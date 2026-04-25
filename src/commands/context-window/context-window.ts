import type { LocalCommandCall } from '../../types/command.js'
import { getGlobalConfig, saveGlobalConfig } from '../../utils/config.js'
import { getGlobalCompatProvider } from '../../utils/customApiStorage.js'

const MIN_CONTEXT_WINDOW = 1_000
const MAX_CONTEXT_WINDOW = 2_000_000

function isCompatProviderActive(): boolean {
  const customConfig = getGlobalConfig().customApiEndpoint
  const provider =
    customConfig?.provider ?? getGlobalCompatProvider(customConfig?.baseURL)
  return provider === 'openai' || provider === 'gemini'
}

export const call: LocalCommandCall = async (args, _context) => {
  const trimmed = args.trim()

  if (!trimmed) {
    const current = getGlobalConfig().customApiEndpoint?.contextWindow
    if (typeof current === 'number' && current > 0) {
      return {
        type: 'text',
        value: `Custom context window: ${current.toLocaleString()} tokens. Run \`/context-window clear\` to reset.`,
      }
    }
    return {
      type: 'text',
      value: `No custom context window set. Usage: /context-window <tokens>${
        isCompatProviderActive()
          ? ''
          : '\n(Note: this override only applies to OpenAI/Gemini compat endpoints.)'
      }`,
    }
  }

  if (trimmed === 'clear' || trimmed === 'reset') {
    saveGlobalConfig(current => ({
      ...current,
      customApiEndpoint: {
        ...current.customApiEndpoint,
        contextWindow: undefined,
      },
    }))
    return { type: 'text', value: 'Cleared custom context window override.' }
  }

  const parsed = Number.parseInt(trimmed, 10)
  if (!Number.isFinite(parsed) || parsed < MIN_CONTEXT_WINDOW || parsed > MAX_CONTEXT_WINDOW) {
    return {
      type: 'text',
      value: `Context window must be an integer between ${MIN_CONTEXT_WINDOW.toLocaleString()} and ${MAX_CONTEXT_WINDOW.toLocaleString()} tokens.`,
    }
  }

  saveGlobalConfig(current => ({
    ...current,
    customApiEndpoint: {
      ...current.customApiEndpoint,
      contextWindow: parsed,
    },
  }))

  return {
    type: 'text',
    value: `Set custom context window to ${parsed.toLocaleString()} tokens.${
      isCompatProviderActive()
        ? ''
        : ' Note: this override only takes effect when the active provider is OpenAI/Gemini compat.'
    }`,
  }
}
