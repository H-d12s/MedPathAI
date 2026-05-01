import { create } from 'zustand'
import { getUserId } from '../api/auth'

export const useUserStore = create((set, get) => ({
  userId: getUserId(),
  profile: null,
  financials: null,
  documents: [],

  setProfile: (profile) => set({ profile }),
  setFinancials: (financials) => set({ financials }),
  setDocuments: (documents) => set({ documents }),
  addDocument: (doc) => set((s) => ({ documents: [...s.documents, doc] })),

  // Derived
  isProfileComplete: () => !!get().profile,
  hasFinancials: () => !!get().financials,
}))