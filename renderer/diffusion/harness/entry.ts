/* Headless render harness: mirrors apps/web/src/engine/capture.ts of diffusionstudio/editor (MPL-2.0) without an editor world.
 * window.__render(code, cfg) mounts a compiled composition bundle in a fresh runtime world and encodes its scene; the file's bytes are streamed to `window.__chunk`. */
import { mount } from '@diffusionstudio/reconciler';
import { ChildOf, FramePromises, FrameRate, Fonts, Library, Mode, RenderSurface, Root, createRuntimeWorld, disposeDecoders, resetCamera } from '@diffusionstudio/runtime';
import { createEncoder } from '@diffusionstudio/encoder';
import { AssetLibrary } from '@diffusionstudio/assets';

// The library only ever resolves remote URLs here (render.mjs serves every file over http), so its file system is empty.
const emptyFs: any = { readManifest: async () => null, writeManifest: async () => {}, list: async () => [], stat: async () => null,
	file: async (s: string) => { throw new Error('no local files: ' + s); }, write: async () => {}, remove: async () => {} };

declare const window: any;

window.__render = async (code: string, cfg: { fps?: number; videoBitrate?: number; resolution?: number }) => {
	const canvas = document.createElement('canvas');
	canvas.width = 2; canvas.height = 2;
	canvas.style.cssText = 'position:fixed;left:0;top:0;z-index:-9999;opacity:0;will-change:opacity;pointer-events:none;';
	document.body.appendChild(canvas);
	const world = createRuntimeWorld('aikyam');
	world.set(Mode, { value: 'offline-video' });
	world.set(Library, new AssetLibrary(emptyFs));
	world.set(Fonts, { list: [] });
	world.set(RenderSurface, { canvas, ctx: canvas.getContext('2d'), resolution: 1 } as any);
	world.set(FrameRate, { value: cfg.fps ?? 30 });
	world.set(FramePromises, { list: [] });
	const mounted = mount(code, world);
	resetCamera(world);                                                                 // the plain view an encode draws with (the editor's last camera is no part of the composition)
	try {
		for (const root of world.query(ChildOf(world.get(Root)!))) void root;          // the stage holds one scene
		const enc = await createEncoder(world, { format: 'mp4', video: { codec: 'avc', bitrate: cfg.videoBitrate ?? 8e6, resolution: cfg.resolution ?? 1080 }, audio: { enabled: false },
			onProgress: (p: any) => window.__progress?.(p.progress / Math.max(p.total, 1)) });
		const res: any = await enc.render();
		if (res.type !== 'success' || !res.data) throw new Error(res.type === 'error' ? String(res.error?.message ?? res.error) : 'render ' + res.type);
		const buf = new Uint8Array(await res.data.arrayBuffer()); const N = 1 << 20;
		for (let i = 0; i < buf.length; i += N) { let s = ''; const part = buf.subarray(i, i + N); for (let j = 0; j < part.length; j += 32768) s += String.fromCharCode(...part.subarray(j, j + 32768)); await window.__chunk(btoa(s)); }
		return { bytes: buf.length };
	} finally { mounted.dispose(); canvas.remove(); try { disposeDecoders(world, world.get(Root)!); } catch {} world.destroy(); }
};
window.__ready = true;
