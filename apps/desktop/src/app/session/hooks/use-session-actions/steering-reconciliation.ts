import { textWithoutReferenceLines } from '@/components/assistant-ui/reference-kinds'
import { type ChatMessage, chatMessageText, sameAttachmentTurn } from '@/lib/chat-messages'

import { conflictingTranscriptIdentity } from './pending-turn-identity'

// Only typed steering rows may unwrap the model's delivery envelope. Ordinary
// prompts containing a lookalike marker remain literal text.
const steeringText = (message: ChatMessage): string => {
  const text = textWithoutReferenceLines(chatMessageText(message))

  return message.steering
    ? (
        text.match(/^\[OUT-OF-BAND USER MESSAGE[^\]]*\]\s*([\s\S]*?)\s*\[\/OUT-OF-BAND USER MESSAGE\]$/)?.[1] ?? text
      ).trim()
    : text
}

/** Pair occurrences after the acknowledged boundary, consuming each receipt once. */
export function acknowledgedSteeringMessages(candidates: ChatMessage[], local: ChatMessage[]): Set<string> {
  const acknowledged = new Set<string>()
  let cursor = 0

  for (const message of local) {
    if (message.role !== 'user') {
      continue
    }

    const index = candidates.findIndex(
      (candidate, at) =>
        at >= cursor &&
        !conflictingTranscriptIdentity(message, candidate) &&
        (!candidate.steering || message.steering === true) &&
        (steeringText(candidate) === steeringText(message) || sameAttachmentTurn(candidate, message))
    )

    if (index < 0) {
      continue
    }

    cursor = index + 1

    if (message.steering) {
      acknowledged.add(message.id)
    }
  }

  return acknowledged
}
