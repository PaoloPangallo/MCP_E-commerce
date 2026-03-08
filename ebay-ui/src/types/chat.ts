import type { SearchBlock } from "./search"
import type { TimedEntity } from "./common"

export interface Message extends TimedEntity {
  role: "user" | "assistant"
  content: string
}

export type ChatEntry =
  | { type: "message"; msg: Message }
  | { type: "search"; search: SearchBlock }