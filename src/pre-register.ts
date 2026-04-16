import { PROVIDER_NAME } from './models'

/**
 * Determine which model IDs must be pre-registered synchronously so they
 * exist in the registry before Pi computes its initial scope.
 */
export function collectPreRegisterModels(settings: unknown): string[] {
  if (!settings || typeof settings !== 'object') return []
  const s = settings as Record<string, unknown>
  const ids = new Set<string>()
  const prefix = `${PROVIDER_NAME}/`

  const enabled = Array.isArray(s.enabledModels) ? s.enabledModels : []
  for (const entry of enabled) {
    if (typeof entry === 'string' && entry.startsWith(prefix)) {
      ids.add(entry.slice(prefix.length))
    }
  }

  if (
    s.defaultProvider === PROVIDER_NAME &&
    typeof s.defaultModel === 'string' &&
    s.defaultModel.length > 0
  ) {
    ids.add(s.defaultModel)
  }

  return [...ids]
}
