import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { homedir } from 'node:os'
import { PROVIDER_NAME } from './models'

const MODELS_JSON_PATH = join(homedir(), '.pi', 'agent', 'models.json')

/**
 * Read user-defined models for this provider from ~/.pi/agent/models.json.
 * Returns null if none are defined, allowing fallback to VERTEX_MODELS.
 * Only reads models scoped to the vertex-anthropic provider;
 * other providers in the same file are unaffected.
 */
export function readModelsFromJson(): any[] | null {
  try {
    const raw = readFileSync(MODELS_JSON_PATH, 'utf-8')
    const parsed = JSON.parse(raw)
    const models = parsed?.providers?.[PROVIDER_NAME]?.models
    return Array.isArray(models) && models.length > 0 ? models : null
  } catch {
    return null
  }
}
