import { create } from "zustand"

type SidebarStore = {
  mobileOpen: boolean
  setMobileOpen: (open: boolean) => void
}

export const useSidebarStore = create<SidebarStore>((set) => ({
  mobileOpen: false,
  setMobileOpen: (open) => set({ mobileOpen: open }),
}))
