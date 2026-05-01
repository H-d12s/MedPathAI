import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import AppShell from './components/AppShell'
import Registration from './screens/Registration'
import Documents from './screens/Documents'
import Chat from './screens/Chat'
import { isRegistered } from './api/auth'

// Guard: redirect to /register if not yet registered
function RequireProfile({ children }) {
  return isRegistered() ? children : <Navigate to="/register" replace />
}

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<AppShell />}>
          <Route path="/register" element={<Registration />} />
          <Route
            path="/documents"
            element={<RequireProfile><Documents /></RequireProfile>}
          />
          <Route
            path="/chat"
            element={<RequireProfile><Chat /></RequireProfile>}
          />
          <Route path="*" element={<Navigate to="/register" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}