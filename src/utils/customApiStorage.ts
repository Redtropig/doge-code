import { getSecureStorage } from './secureStorage/index.js'

export type OpenAICompatMode = 'chat_completions' | 'responses'

export type CustomApiProvider = 'anthropic' | 'openai' | 'gemini'

export function getGlobalCompatProvider(baseURL?: string): CustomApiProvider {
  const url = baseURL || process.env.ANTHROPIC_BASE_URL || ''
  if (url.includes('integrate.api.nvidia.com')) {
    return 'openai'
  }
  return process.env.CLAUDE_CODE_COMPATIBLE_API_PROVIDER === 'openai'
    ? 'openai'
    : process.env.CLAUDE_CODE_COMPATIBLE_API_PROVIDER === 'gemini'
      ? 'gemini'
      : 'anthropic'
}

export type CustomApiStorageData = {
  provider?: CustomApiProvider
  openaiCompatMode?: OpenAICompatMode
  baseURL?: string
  apiKey?: string
  model?: string
  savedModels?: string[]
}

const CUSTOM_API_STORAGE_KEY = 'customApiEndpoint'

export function readCustomApiStorage(): CustomApiStorageData {
  const storage = getSecureStorage() as {
    read?: () => Record<string, unknown> | null
    update?: (data: Record<string, unknown>) => { success: boolean }
  }
  const data = storage.read?.() ?? {}
  const raw = data[CUSTOM_API_STORAGE_KEY]
  if (!raw || typeof raw !== 'object') return {}
  const value = raw as Record<string, unknown>
  const provider =
    value.provider === 'openai' || value.provider === 'anthropic' || value.provider === 'gemini'
      ? value.provider
      : undefined
  const openaiCompatMode =
    value.openaiCompatMode === 'chat_completions' || value.openaiCompatMode === 'responses'
      ? value.openaiCompatMode
      : provider === 'openai'
        ? 'chat_completions'
        : undefined

  return {
    provider,
    openaiCompatMode,
    baseURL: typeof value.baseURL === 'string' ? value.baseURL : undefined,
    apiKey: typeof value.apiKey === 'string' ? value.apiKey : undefined,
    model: typeof value.model === 'string' ? value.model : undefined,
    savedModels: Array.isArray(value.savedModels)
      ? value.savedModels.filter((item): item is string => typeof item === 'string')
      : [],
  }
}

export function writeCustomApiStorage(next: CustomApiStorageData): void {
  const storage = getSecureStorage() as {
    read?: () => Record<string, unknown> | null
    update?: (data: Record<string, unknown>) => { success: boolean }
  }
  const current = storage.read?.() ?? {}
  storage.update?.({
    ...current,
    customApiEndpoint: next,
  })
}

export function clearCustomApiStorage(): void {
  const storage = getSecureStorage() as {
    read?: () => Record<string, unknown> | null
    update?: (data: Record<string, unknown>) => { success: boolean }
  }
  const current = storage.read?.() ?? {}
  const { customApiEndpoint: _, ...rest } = current
  storage.update?.(rest)
}
