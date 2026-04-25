import type { Command } from '../../commands.js'

export default {
  type: 'local',
  name: 'context-window',
  description:
    'View or set the context window size (tokens) used for OpenAI/Gemini compat endpoints. Usage: /context-window [tokens|clear]',
  supportsNonInteractive: true,
  load: () => import('./context-window.js'),
} satisfies Command
