import { create } from "zustand"

export interface SidebarState {
  mobileOpen: boolean
  isCollapsed: boolean
  width: number
  setMobileOpen: (open: boolean) => void
  setIsCollapsed: (collapsed: boolean) => void
  setWidth: (width: number) => void
}

export const useSidebarStore = create<SidebarState>((set) => ({
  mobileOpen: false,
  isCollapsed: false,
  width: 260,
  setMobileOpen: (open) => set({ mobileOpen: open }),
  setIsCollapsed: (collapsed) => set({ isCollapsed: collapsed }),
  setWidth: (width) => set({ width }),
}))
