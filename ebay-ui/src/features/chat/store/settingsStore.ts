import { create } from "zustand"
import { apiFetch } from "../../../api/apiClient"

export interface UserSettings {
  theme: 'light' | 'dark'
  conversationTone: 'neutral' | 'amichevole' | 'professionale'
  customInstructions: string
}

interface SettingsState {
  settings: UserSettings
  isOpen: boolean
  isSaving: boolean
  
  setOpen: (open: boolean) => void
  updateLocalSettings: (partial: Partial<UserSettings>) => void
  saveSettingsToBackend: (token: string, newSettings: UserSettings) => Promise<void>
  loadSettingsFromAuth: (theme?: string, tone?: string, instructions?: string) => void
}

export const useSettingsStore = create<SettingsState>((set, get) => ({
  settings: {
    theme: 'light',
    conversationTone: 'neutral',
    customInstructions: ''
  },
  isOpen: false,
  isSaving: false,

  setOpen: (open) => set({ isOpen: open }),

  updateLocalSettings: (partial) => {
    set((state) => ({
      settings: { ...state.settings, ...partial }
    }))
  },

  loadSettingsFromAuth: (theme, tone, instructions) => {
    set((state) => ({
      settings: {
        theme: (theme as 'light'|'dark') || 'light',
        conversationTone: (tone as any) || 'neutral',
        customInstructions: instructions || ''
      }
    }))
    
    // Al caricamento, applichiamo il tema al body
    const isDark = theme === 'dark'
    if (isDark) {
      document.body.classList.add('dark-mode')
    } else {
      document.body.classList.remove('dark-mode')
    }
  },

  saveSettingsToBackend: async (token, newSettings) => {
    set({ isSaving: true })
    try {
      const resp = await apiFetch<any>('/auth/me/preferences', {
        method: 'PATCH',
        body: JSON.stringify({
          theme: newSettings.theme,
          conversation_tone: newSettings.conversationTone,
          custom_instructions: newSettings.customInstructions
        })
      })

      const data = resp;
      
      set({ 
        settings: {
          theme: data.theme || 'light',
          conversationTone: data.conversation_tone || 'neutral',
          customInstructions: data.custom_instructions || ''
        },
        isSaving: false 
      })

      // Applica il tema
      if (data.theme === 'dark') {
        document.body.classList.add('dark-mode')
      } else {
        document.body.classList.remove('dark-mode')
      }

    } catch (err) {
      console.error(err)
      set({ isSaving: false })
      throw err
    }
  }
}))
