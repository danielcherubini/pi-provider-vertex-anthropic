import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { homedir } from 'node:os'
import type { ExtensionAPI } from '@mariozechner/pi-coding-agent'

import { findGoogleCloudCliPath, getAccessToken, getGoogleCloudCliToken } from './auth'
import { buildEndpointHost, resolveConfig } from './config'
import { API_NAME, PROVIDER_NAME, VERTEX_MODELS, type VertexModel } from './models'
import { streamVertexAnthropic } from './vertex-api'
import { exec, spawn } from './shell'
import { collectPreRegisterModels } from './pre-register'

const SETTINGS_PATH = join(homedir(), '.pi', 'agent', 'settings.json')

const REGION_CHOICES: Record<string, string> = {
  '1': 'global',
  '2': 'us-east5',
  '3': 'us-central1',
  '4': 'europe-west1',
  '5': 'asia-southeast1',
}

/**
 * Map Anthropic API model info to our VertexModel format.
 */
function mapAnthropicModelToVertex(
  anthropicModel: {
    id: string
    display_name: string
    max_input_tokens: number
    max_tokens: number
    capabilities: {
      thinking?: { supported: boolean; types?: string[] }
      images?: { supported: boolean }
      tools?: { supported: boolean }
      structured?: { supported: boolean }
    }
  },
  pricing: VertexModel['cost'],
): VertexModel {
  const id = anthropicModel.id
  // Strip the @timestamp suffix for Vertex compatibility if present
  const vertexId = id.includes('@') ? id.split('@')[0] : id

  return {
    id: vertexId,
    name: `${anthropicModel.display_name} (Vertex)`,
    reasoning: anthropicModel.capabilities.thinking?.supported ?? false,
    input: [
      'text',
      ...(anthropicModel.capabilities.images?.supported ? ['image'] : []),
    ] as ('text' | 'image')[],
    contextWindow: anthropicModel.max_input_tokens,
    maxTokens: anthropicModel.max_tokens,
    cost: pricing,
  }
}

/**
 * Fetch available Claude models from Vertex AI's Anthropic endpoint.
 * Returns undefined if credentials aren't configured yet (first-time setup).
 */
async function fetchVertexModels(): Promise<{ models: VertexModel[]; project?: string; region?: string } | undefined> {
  const config = resolveConfig()

  if (!config.project) {
    // Credentials not configured yet — first-time /login flow hasn't run
    return undefined
  }

  const host = buildEndpointHost(config.region)
  const url = `https://${host}/v1/projects/${config.project}/locations/${config.region}/publishers/anthropic/models`

  try {
    const token = await getAccessToken()
    const response = await fetch(url, {
      headers: { Authorization: `Bearer ${token}` },
      signal: AbortSignal.timeout(5000),
    })

    if (!response.ok) {
      console.warn(`[vertex-anthropic] Failed to list models (${response.status})`)
      return undefined
    }

    const data = await response.json()
    const models: VertexModel[] = []

    for (const m of data.models || []) {
      // Only include Anthropic Claude models (skip Gemini, etc.)
      if (!m.id.startsWith('claude-')) continue

      const pricing = VERTEX_PRICING[m.id] ?? VERTEX_PRICING[m.id.split('@')[0]] ?? DEFAULT_PRICING
      const model = mapAnthropicModelToVertex(m, pricing)
      models.push(model)
    }

    // Attach project/region to each model so streamVertexAnthropic can find them
    for (const m of models) {
      ;(m as any).project = config.project
      ;(m as any).region = config.region
    }

    return { models, project: config.project, region: config.region }
  } catch (error) {
    const msg = error instanceof Error ? error.message : 'Unknown error'
    console.warn(`[vertex-anthropic] Model discovery failed: ${msg}`)
    return undefined
  }
}

// Pricing lookup by model ID
const VERTEX_PRICING: Record<string, VertexModel['cost']> = {
  // Opus
  'claude-opus-4-6': { input: 5, output: 25, cacheRead: 0.5, cacheWrite: 6.25 },
  'claude-opus-4-5@20251101': { input: 15, output: 75, cacheRead: 1.5, cacheWrite: 18.75 },
  // Sonnet
  'claude-sonnet-4-5@20250929': { input: 3, output: 15, cacheRead: 0.3, cacheWrite: 3.75 },
  // Haiku
  'claude-haiku-4-5@20251001': { input: 1, output: 5, cacheRead: 0.1, cacheWrite: 1.25 },
  // Claude 3.x
  'claude-3-5-sonnet-v2@20241022': { input: 3, output: 15, cacheRead: 0.3, cacheWrite: 3.75 },
  'claude-3-5-sonnet@20240620': { input: 3, output: 15, cacheRead: 0.3, cacheWrite: 3.75 },
  'claude-3-5-haiku@20241022': { input: 0.8, output: 4, cacheRead: 0.08, cacheWrite: 1 },
  'claude-3-opus@20240229': { input: 15, output: 75, cacheRead: 1.5, cacheWrite: 18.75 },
  'claude-3-sonnet@20240229': { input: 3, output: 15, cacheRead: 0.3, cacheWrite: 3.75 },
  'claude-3-haiku@20240307': { input: 0.25, output: 1.25, cacheRead: 0.03, cacheWrite: 0.3 },
}

const DEFAULT_PRICING: VertexModel['cost'] = { input: 3, output: 15, cacheRead: 0.3, cacheWrite: 3.75 }

export default async function(pi: ExtensionAPI) {
  const config = resolveConfig()
  const googleCloudCli = findGoogleCloudCliPath()

  // Try dynamic model discovery from Vertex AI
  const discovered = await fetchVertexModels()
  const models = discovered?.models ?? VERTEX_MODELS

  // Synchronous pre-registration to prevent race condition with scoped models
  try {
    const raw = readFileSync(SETTINGS_PATH, 'utf-8')
    const settings = JSON.parse(raw)
    const modelIds = collectPreRegisterModels(settings)

    if (modelIds.length > 0) {
      pi.registerProvider(PROVIDER_NAME, {
        baseUrl: `https://${buildEndpointHost(config.region)}`,
        api: API_NAME,
        apiKey: 'vertex-anthropic',
        models: modelIds.map((id) => {
          const known = models.find((m) => m.id === id)
          return known || {
            id,
            name: id,
            reasoning: false,
            input: ['text', 'image'] as ('text' | 'image')[],
            contextWindow: 200000,
            maxTokens: 8192,
            cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
          }
        }),
        streamSimple: streamVertexAnthropic,
      })
    }
  } catch {
    // Non-fatal: pre-registration is best-effort
  }

  // Full provider registration with all models and /login support
  pi.registerProvider(PROVIDER_NAME, {
    baseUrl: `https://${buildEndpointHost(config.region)}`,
    api: API_NAME,
    apiKey: 'vertex-anthropic',

    oauth: {
      name: 'Google Cloud Vertex AI',

      async login(callbacks) {
        callbacks.onProgress?.('Setting up Google Cloud Vertex AI...')

        // Check if gcloud is installed
        try {
          const gcloudPath = googleCloudCli || findGoogleCloudCliPath()
          if (!gcloudPath) throw new Error('gcloud not found')
          spawn(gcloudPath, ['version'], { stdio: 'ignore', timeout: 2000 })
        } catch {
          const install = await callbacks.onPrompt({
            message:
              'gcloud CLI not found. Install Google Cloud SDK?\n\n' +
              'https://cloud.google.com/sdk/docs/install\n\n(y/n)',
          })

          if (install?.toLowerCase() === 'y') {
            callbacks.onAuth({
              url: 'https://cloud.google.com/sdk/docs/install',
              instructions: 'Install the Google Cloud SDK, then run /login again',
            })
            throw new Error('Please install gcloud CLI and run /login again')
          }
          throw new Error(
            'gcloud CLI required. Install from: https://cloud.google.com/sdk/docs/install',
          )
        }

        // Check authentication
        callbacks.onProgress?.('Checking gcloud authentication...')
        let needsAuth = false
        try {
          const gcloudPath = googleCloudCli || findGoogleCloudCliPath()
          if (!gcloudPath) throw new Error('gcloud not found')
          const token = exec(gcloudPath, ['auth', 'print-access-token'], { timeout: 5000 })
          needsAuth = !token || token.includes('ERROR') || token.length < 20
        } catch {
          needsAuth = true
        }

        if (needsAuth) {
          const doAuth = await callbacks.onPrompt({
            message: "Not authenticated with gcloud. Run 'gcloud auth login' now? (y/n)",
          })

          if (doAuth?.toLowerCase() === 'y') {
            callbacks.onProgress?.('Running gcloud auth login (a browser window will open)...')
            try {
              const gcloudPath = googleCloudCli || findGoogleCloudCliPath()
              if (!gcloudPath) throw new Error('gcloud not found')
              spawn(gcloudPath, ['auth', 'login'], { stdio: 'inherit' })
            } catch {
              throw new Error('Authentication failed. Please try: gcloud auth login')
            }
          } else {
            throw new Error('Authentication required. Run: gcloud auth login')
          }
        }

        // Configure project
        callbacks.onProgress?.('Configuring project...')
        let project =
          process.env.ANTHROPIC_VERTEX_PROJECT_ID || process.env.GOOGLE_CLOUD_PROJECT

        if (!project) {
          // Try current gcloud default
          try {
            const gcloudPath = googleCloudCli || findGoogleCloudCliPath()
            if (!gcloudPath) throw new Error('gcloud not found')
            const currentProject = exec(gcloudPath, ['config', 'get-value', 'project'], {
              timeout: 5000,
            })

            if (currentProject && currentProject !== '(unset)') {
              const use = await callbacks.onPrompt({
                message: `Use current project '${currentProject}'? (y/n)`,
              })
              if (use?.toLowerCase() === 'y') {
                project = currentProject
              }
            }
          } catch {
            // Ignore
          }

          if (!project) {
            let projectPrompt = 'Enter GCP project ID:'
            try {
              const gcloudPath = googleCloudCli || findGoogleCloudCliPath()
              if (!gcloudPath) throw new Error('gcloud not found')
              const projects = exec(
                gcloudPath,
                ['projects', 'list', '--format=value(projectId)'],
                { timeout: 10000 },
              )
                .split('\n')
                .filter((p) => p && p !== '(unset)')

              if (projects.length > 0 && projects.length < 20) {
                projectPrompt = `Available projects:\n${projects.map((p) => `  - ${p}`).join('\n')}\n\nEnter project ID:`
              }
            } catch {
              // Ignore
            }

            const projectInput = await callbacks.onPrompt({ message: projectPrompt })
            if (!projectInput || projectInput.trim() === '') {
              throw new Error('Project ID required')
            }
            project = projectInput.trim()

            try {
              const gcloudPath = googleCloudCli || findGoogleCloudCliPath()
              if (gcloudPath) {
                spawn(gcloudPath, ['config', 'set', 'project', project], { stdio: 'ignore' })
              }
            } catch {
              // Non-fatal
            }
          }
        }

        // Also set in process env for this session
        process.env.ANTHROPIC_VERTEX_PROJECT_ID = project

        // Configure region
        callbacks.onProgress?.('Configuring region...')
        let region =
          process.env.CLOUD_ML_REGION ||
          process.env.VERTEX_LOCATION ||
          process.env.VERTEXAI_LOCATION

        if (!region) {
          const regionChoice = await callbacks.onPrompt({
            message:
              'Select region:\n\n' +
              '  1. global (recommended for latest models)\n' +
              '  2. us-east5\n' +
              '  3. us-central1\n' +
              '  4. europe-west1\n' +
              '  5. asia-southeast1\n\n' +
              'Enter 1-5 or custom region name:',
          })

          region =
            REGION_CHOICES[regionChoice || ''] || regionChoice?.trim() || 'global'
          process.env.CLOUD_ML_REGION = region
        }

        // Check if Vertex AI API is enabled
        callbacks.onProgress?.('Checking Vertex AI API...')
        try {
          const gcloudPath = googleCloudCli || findGoogleCloudCliPath()
          if (!gcloudPath) throw new Error('gcloud not found')
          const enabled = exec(
            gcloudPath,
            [
              'services',
              'list',
              '--enabled',
              '--filter=name:aiplatform.googleapis.com',
              '--format=value(name)',
            ],
          )

          if (!enabled) {
            const enable = await callbacks.onPrompt({
              message: 'Vertex AI API not enabled. Enable it now? (y/n)',
            })

            if (enable?.toLowerCase() === 'y') {
              callbacks.onProgress?.('Enabling Vertex AI API (this may take a minute)...')
              spawn(gcloudPath, ['services', 'enable', 'aiplatform.googleapis.com'], {
                stdio: 'inherit',
              })
              callbacks.onProgress?.('Vertex AI API enabled!')
            } else {
              callbacks.onProgress?.(
                'API not enabled. Enable it manually:\n' +
                `  gcloud services enable aiplatform.googleapis.com --project=${project}`,
              )
            }
          }
        } catch {
          callbacks.onProgress?.(
            'Could not check API status. If requests fail, enable it manually:\n' +
            `  gcloud services enable aiplatform.googleapis.com --project=${project}`,
          )
        }

        // Test authentication
        callbacks.onProgress?.('Testing authentication...')
        try {
          const gcloudPath = googleCloudCli || findGoogleCloudCliPath()
          if (!gcloudPath) throw new Error('gcloud not found')
          const token = exec(gcloudPath, ['auth', 'print-access-token'], { timeout: 5000 })

          if (!token || token.length < 20) {
            throw new Error('Invalid token')
          }
        } catch {
          throw new Error('Authentication test failed. Please run: gcloud auth login')
        }

        callbacks.onProgress?.(
          `Configured successfully!\n` +
          `Project: ${project}\n` +
          `Region: ${region}\n` +
          `Settings persisted to ~/.pi/agent/auth.json.\n` +
          `If authentication fails later, run: gcloud auth login`,
        )

        return {
          refresh: Date.now().toString(),
          access: 'gcloud',
          expires: Date.now() + 1000 * 60 * 60 * 24 * 365,
          project,
          region,
        }
      },

      async refreshToken(credentials) {
        return { ...credentials, refresh: Date.now().toString() }
      },

      getApiKey() {
        try {
          return getGoogleCloudCliToken(googleCloudCli)
        } catch (error) {
          const msg = error instanceof Error ? error.message : 'Unknown error'
          throw new Error(`Failed to get access token: ${msg}\n\nRun: gcloud auth login`)
        }
      },
    },

    models: models,
    streamSimple: streamVertexAnthropic,
  })

}
