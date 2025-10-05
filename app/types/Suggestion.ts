import type { Position, RawPosition } from './Position'

/**
 * Type definition for a suggestion request.
 */
export interface SuggestionRequest {
  /** The name of the model to use. */
  model: string
  /** A unique ID for the request. */
  request_id: string
}

/**
 * A raw suggested position object as received from the server.
 */
export interface RawSuggestedPosition {
  /** The suggested position. */
  position: RawPosition
  /** The confidence score of the suggestion. */
  confidence: number
}

/**
 * Type definition for a suggestion response.
 */
export interface SuggestionResponse {
  /** The ID of the request, which should match the request_id of the corresponding SuggestionRequest. */
  request_id: string
  /** A list of suggested positions. */
  positions: RawSuggestedPosition[]
}

/**
 * A single suggested position with its confidence score.
 */
export interface SuggestedPosition {
  /** The suggested position. */
  position: Position
  /** The confidence score of the suggestion. */
  confidence: number
}

/**
 * The response format from the model.
 */
export interface ModelResponse {
  /** A list of suggested positions. */
  positions: SuggestedPosition[]
}
