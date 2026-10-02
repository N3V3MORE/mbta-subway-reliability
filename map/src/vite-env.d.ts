/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Set by the Pages workflow, which deploys reports/ next to the app. Unset in local and docs/ builds. */
  readonly VITE_REPORTS?: string
}
