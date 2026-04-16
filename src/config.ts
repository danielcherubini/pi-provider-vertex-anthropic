import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { homedir } from 'node:os'

const AUTH_PATH = join(homedir(), '.pi', 'agent', 'auth.json')
const DEFAULT_REGION = 'us-east5'

export interface PersistedCredentials {
  project?: string
  region?: string
}

export interface VertexConfig {
  project: string
  region: string
}

/**
 * Read persisted credentials from ~/.pi/agent/auth.json.
 * These are stored by the /login interactive flow.
 */
export function getPersistedCredentials(): PersistedCredentials {
  try {
    const data = JSON.parse(readFileSync(AUTH_PATH, 'utf-8'))
    const cred = data['vertex-anthropic']
    if (cred?.type === 'oauth') {
      return { project: cred.project, region: cred.region }
    }
  } catch {
    // No persisted credentials available
  }
  return {}
}

/**
 * Resolve the GCP project ID from environment variables and persisted credentials.
 *
 * Priority:
 *  1. ANTHROPIC_VERTEX_PROJECT_ID (Claude CLI)
 *  2. GOOGLE_CLOUD_PROJECT (Opencode / standard GCP)
 *  3. Persisted credentials from auth.json
 */
export function resolveProject(persisted?: PersistedCredentials): string | undefined {
  return (
    process.env.ANTHROPIC_VERTEX_PROJECT_ID ||
    process.env.GOOGLE_CLOUD_PROJECT ||
    persisted?.project ||
    undefined
  )
}

/**
 * Resolve the Vertex AI region from environment variables and persisted credentials.
 *
 * Priority:
 *  1. CLOUD_ML_REGION (Claude CLI)
 *  2. VERTEX_LOCATION (Opencode)
 *  3. VERTEXAI_LOCATION (Opencode alternative)
 *  4. Persisted credentials from auth.json
 *  5. Default: us-east5
 */
export function resolveRegion(persisted?: PersistedCredentials): string {
  return (
    process.env.CLOUD_ML_REGION ||
    process.env.VERTEX_LOCATION ||
    process.env.VERTEXAI_LOCATION ||
    persisted?.region ||
    DEFAULT_REGION
  )
}

/**
 * Build the Vertex AI endpoint hostname.
 * The `global` region uses `aiplatform.googleapis.com` without a region prefix.
 */
export function buildEndpointHost(region: string): string {
  return region === 'global'
    ? 'aiplatform.googleapis.com'
    : `${region}-aiplatform.googleapis.com`
}

/**
 * Build the full Vertex AI streamRawPredict URL for a given model.
 */
export function buildStreamUrl(region: string, project: string, modelId: string): string {
  const host = buildEndpointHost(region)
  return `https://${host}/v1/projects/${project}/locations/${region}/publishers/anthropic/models/${modelId}:streamRawPredict`
}

/**
 * Resolve full config from env vars and persisted credentials.
 */
export function resolveConfig(): VertexConfig {
  const persisted = getPersistedCredentials()
  return {
    project: resolveProject(persisted) || '',
    region: resolveRegion(persisted),
  }
}
