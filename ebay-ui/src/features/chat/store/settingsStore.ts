import { create } from "zustand"
import { apiFetch } from "../../../api/apiClient"

export interface UserSettings {
  theme: 'light' | 'dark'
  conversationTone: 'neutral' | 'amichevole' | 'professionale'
  customInstructions: string
  favoriteBrands: string
  pricePreference: string
  contextualBudgets?: string
}

interface SettingsState {
  settings: UserSettings
  isOpen: boolean
  isSaving: boolean
  
  setOpen: (open: boolean) => void
  updateLocalSettings: (partial: Partial<UserSettings>) => void
  saveSettingsToBackend: (token: string, newSettings: UserSettings) => Promise<void>
  loadSettingsFromAuth: (theme?: string, tone?: string, instructions?: string, brands?: string, price?: string, budgets?: string) => void
  refreshSettings: () => Promise<void>
}

export const useSettingsStore = create<SettingsState>((set) => ({
  settings: {
    theme: 'light',
    conversationTone: 'neutral',
    customInstructions: '',
    favoriteBrands: '',
    pricePreference: '',
    contextualBudgets: ''
  },
  isOpen: false,
  isSaving: false,

  setOpen: (open) => set({ isOpen: open }),

  updateLocalSettings: (partial) => {
    set((state) => ({
      settings: { ...state.settings, ...partial }
    }))
  },

  loadSettingsFromAuth: (theme, tone, instructions, brands, price, budgets) => {
    set(() => ({
      settings: {
        theme: (theme as 'light'|'dark') || 'light',
        conversationTone: (tone as any) || 'neutral',
        customInstructions: instructions || '',
        favoriteBrands: brands || '',
        pricePreference: price || '',
        contextualBudgets: budgets || ''
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

  saveSettingsToBackend: async (_token, newSettings) => {
    set({ isSaving: true })
    try {
      const resp = await apiFetch<any>('/auth/me/preferences', {
        method: 'PATCH',
        body: JSON.stringify({
          theme: newSettings.theme,
          conversation_tone: newSettings.conversationTone,
          custom_instructions: newSettings.customInstructions,
          favorite_brands: newSettings.favoriteBrands,
          price_preference: newSettings.pricePreference,
          contextual_budgets: newSettings.contextualBudgets
        })
      })

      const data = resp;
      
      set({ 
        settings: {
          theme: data.theme || 'light',
          conversationTone: data.conversation_tone || 'neutral',
          customInstructions: data.custom_instructions || '',
          favoriteBrands: data.favorite_brands || '',
          pricePreference: data.price_preference || '',
          contextualBudgets: data.contextual_budgets || ''
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
  },

  refreshSettings: async () => {
    try {
      const resp = await apiFetch<any>('/auth/me')
      set(() => ({
        settings: {
          theme: resp.theme || 'light',
          conversationTone: resp.conversation_tone || 'neutral',
          customInstructions: resp.custom_instructions || '',
          favoriteBrands: resp.favorite_brands || '',
          pricePreference: resp.price_preference || '',
          contextualBudgets: resp.contextual_budgets || ''
        }
      }))
      
      // Sync theme class
      if (resp.theme === 'dark') {
          document.body.classList.add('dark-mode')
      } else {
          document.body.classList.remove('dark-mode')
      }
    } catch (err) {
      console.warn("Failed to refresh settings:", err)
    }
  }
}))
