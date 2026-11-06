import { configureStore } from '@reduxjs/toolkit'

/* `dashboard.new_chat_crew_mode`: a create that opts in (`crewModeDefault:
 * true`, the person's own New chat entry points) asks the server to start it in
 * Ultra Mode, and only while this dashboard runs Ultra Mode (the preview on, a
 * gateway with Crew Mode). The server decides the rest, so the client never
 * names the conductor itself and a folder's agent pin still wins there. */

const api = vi.hoisted(() => ({
  createChatSlot: vi.fn(),
  chatSlotProject: vi.fn(() => Promise.resolve({ ok: true })),
  deleteChatSlot: vi.fn(() => Promise.resolve({ ok: true })),
  setSlotColor: vi.fn(() => Promise.resolve({ ok: true })),
  dashboardConfig: vi.fn(() => Promise.resolve({ default_memory_mode: 'persistent' })),
}))

vi.mock('../lib/scrollQuiet', () => ({ whenScrollQuiet: () => Promise.resolve() }))
vi.mock('../api/client', () => ({ api }))

import dashboardReducer from './dashboardSlice'
import chatReducer, { createSlot } from './chatSlice'
import { resetCrewModeCatalog, setCrewModeCatalog } from '../lib/crewMode'
import { PREVIEW_CREW_MODE, setPreviewFlag } from '../utils/previewFlags'

// api.createChatSlot(name, agent, model, mode, memory_mode, title, artifact,
//   folder_id, agent_kind, no_session_folder, app_agent_owner, default_folder,
//   crew_mode_default)
const ARG_AGENT = 1
const ARG_CREW_MODE_DEFAULT = 12

const makeStore = () => configureStore({
  reducer: { chat: chatReducer, dashboard: dashboardReducer },
  middleware: g => g({ serializableCheck: false, immutableCheck: false }),
})

const created = async (opts: Parameters<typeof createSlot>[0]) => {
  await makeStore().dispatch(createSlot(opts))
  return api.createChatSlot.mock.calls[0]
}

describe('createSlot new chat Ultra Mode default', () => {
  beforeEach(() => {
    api.createChatSlot.mockReset()
    api.createChatSlot.mockResolvedValue({ key: 'NEW', title: 'New' })
    setPreviewFlag(PREVIEW_CREW_MODE, true)
    setCrewModeCatalog('kirocrew-conductor', ['kirocrew-conductor'])
  })
  afterEach(() => {
    resetCrewModeCatalog()
    localStorage.removeItem(PREVIEW_CREW_MODE)
  })

  it('asks for it on an opt-in create, naming no agent itself', async () => {
    const call = await created({ crewModeDefault: true, defaultFolder: true })
    expect(call[ARG_CREW_MODE_DEFAULT]).toBe(true)
    expect(call[ARG_AGENT]).toBeUndefined()
  })

  it('passes a folder pin through untouched beside the ask', async () => {
    const call = await created({ crewModeDefault: true, agent: 'reviewer', folder_id: 'work' })
    expect(call[ARG_AGENT]).toBe('reviewer')
  })

  it.each([
    ['no opt-in', { defaultFolder: true }],
    ['no options', undefined],
  ])('does not ask with %s', async (_label, opts) => {
    expect((await created(opts))[ARG_CREW_MODE_DEFAULT]).toBeUndefined()
  })

  it('does not ask while the Ultra Mode preview is off', async () => {
    localStorage.removeItem(PREVIEW_CREW_MODE)
    expect((await created({ crewModeDefault: true }))[ARG_CREW_MODE_DEFAULT]).toBeUndefined()
  })

  it('does not ask on a gateway without Crew Mode', async () => {
    setCrewModeCatalog(undefined, undefined)
    expect((await created({ crewModeDefault: true }))[ARG_CREW_MODE_DEFAULT]).toBeUndefined()
  })
})
