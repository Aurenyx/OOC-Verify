import { type ReactNode } from 'react';
import { Link, useLocation } from 'wouter';
import {
  Activity,
  Beaker,
  ChevronRight,
  Database,
  LayoutDashboard,
  Menu,
  ShieldCheck,
  X,
} from 'lucide-react';
import { useState } from 'react';

const navigation = [
  { href: '/', label: 'Overview', icon: LayoutDashboard },
  { href: '/verify', label: 'Verify', icon: ShieldCheck },
  { href: '/history', label: 'History', icon: Activity },
  { href: '/experiments', label: 'Experiments', icon: Beaker },
  { href: '/dataset', label: 'Dataset browser', icon: Database },
];

export function AppShell({ children }: { children: ReactNode }) {
  const [location] = useLocation();
  const [mobileOpen, setMobileOpen] = useState(false);
  const current = navigation.find((item) => item.href === location) ?? navigation.find((item) => item.href !== '/' && location.startsWith(item.href));

  return (
    <div className="app-shell min-h-[100dvh] text-foreground">
      <aside className="fixed inset-y-0 left-0 z-40 hidden w-[248px] flex-col border-r border-sidebar-border bg-sidebar text-sidebar-foreground lg:flex">
        <SidebarContent location={location} />
      </aside>
      {mobileOpen && (
        <div className="fixed inset-0 z-50 lg:hidden">
          <button
            type="button"
            aria-label="Close navigation"
            data-testid="button-close-navigation"
            className="absolute inset-0 bg-slate-950/30 backdrop-blur-sm"
            onClick={() => setMobileOpen(false)}
          />
          <aside className="relative flex h-full w-[276px] flex-col bg-sidebar text-sidebar-foreground shadow-2xl">
            <div className="absolute right-3 top-3">
              <button type="button" aria-label="Close navigation" data-testid="button-close-navigation-panel" className="focus-ring rounded-lg p-2 text-sidebar-foreground/70 hover:bg-sidebar-accent hover:text-white" onClick={() => setMobileOpen(false)}>
                <X className="h-4 w-4" />
              </button>
            </div>
            <SidebarContent location={location} onNavigate={() => setMobileOpen(false)} />
          </aside>
        </div>
      )}
      <div className="lg:pl-[248px]">
        <header className="sticky top-0 z-30 border-b border-border/80 bg-background/85 backdrop-blur-xl">
          <div className="mx-auto flex h-[72px] max-w-[1500px] items-center justify-between px-5 sm:px-8">
            <div className="flex items-center gap-3">
              <button type="button" aria-label="Open navigation" data-testid="button-open-navigation" className="focus-ring rounded-xl border border-border bg-card p-2.5 text-muted-foreground shadow-sm hover:text-foreground lg:hidden" onClick={() => setMobileOpen(true)}>
                <Menu className="h-4 w-4" />
              </button>
              <div className="hidden items-center gap-2 text-sm text-muted-foreground sm:flex">
                <span>OOC-Verify</span>
                <ChevronRight className="h-3.5 w-3.5" />
                <span className="font-medium text-foreground">{current?.label ?? 'Overview'}</span>
              </div>
              <span className="text-sm font-semibold text-foreground sm:hidden">{current?.label ?? 'Overview'}</span>
            </div>
            <div className="flex items-center gap-3">
              <span className="hidden rounded-full border border-accent/70 bg-accent/40 px-3 py-1.5 font-mono text-[10px] uppercase tracking-[0.16em] text-accent-foreground sm:inline-flex">Pipeline Live</span>
              <div className="flex h-9 w-9 items-center justify-center rounded-full bg-gradient-to-br from-indigo-500 to-violet-500 text-xs font-bold text-white shadow-sm">RS</div>
            </div>
          </div>
        </header>
        <main className="mx-auto max-w-[1500px] px-5 py-8 sm:px-8 lg:py-10">{children}</main>
      </div>
    </div>
  );
}

function SidebarContent({ location, onNavigate }: { location: string; onNavigate?: () => void }) {
  return (
    <>
      <div className="flex h-[72px] items-center gap-3 border-b border-sidebar-border px-6">
        <div className="relative flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-violet-400 via-indigo-500 to-cyan-400 shadow-lg shadow-indigo-950/20">
          <div className="h-3 w-3 rounded-full border-2 border-white/90" />
          <div className="absolute h-6 w-6 rounded-full border border-white/30" />
        </div>
        <div>
          <div className="text-[13px] font-extrabold tracking-tight text-white leading-tight">OOC-<span className="text-violet-300">Verify</span></div>
          <div className="font-mono text-[9px] uppercase tracking-[0.14em] text-sidebar-foreground/60">Multimodal Research</div>
        </div>
      </div>
      <div className="flex flex-1 flex-col px-4 py-7">
        <p className="px-3 text-[10px] font-semibold uppercase tracking-[0.18em] text-sidebar-foreground/40">Workspace</p>
        <nav aria-label="Primary navigation" className="mt-3 space-y-1">
          {navigation.map((item) => {
            const active = item.href === '/' ? location === '/' : location.startsWith(item.href);
            const Icon = item.icon;
            return (
              <Link
                key={item.href}
                href={item.href}
                data-testid={`link-nav-${item.label.toLowerCase().replaceAll(' ', '-')}`}
                onClick={onNavigate}
                className={`focus-ring group flex items-center gap-3 rounded-xl px-3 py-3 text-sm transition-colors ${active ? 'bg-sidebar-accent text-white shadow-inner' : 'text-sidebar-foreground/65 hover:bg-sidebar-accent/70 hover:text-white'}`}
              >
                <Icon className={`h-[17px] w-[17px] ${active ? 'text-violet-300' : 'text-sidebar-foreground/55 group-hover:text-violet-300'}`} />
                <span className="font-semibold">{item.label}</span>
                {active && <span className="ml-auto h-1.5 w-1.5 rounded-full bg-cyan-300" />}
              </Link>
            );
          })}
        </nav>
        <div className="mt-auto rounded-2xl border border-sidebar-border bg-sidebar-accent/45 p-4">
          <div className="flex items-center justify-between">
            <span className="font-mono text-[9px] uppercase tracking-[0.15em] text-sidebar-foreground/45">Environment</span>
            <span className="h-2 w-2 rounded-full bg-cyan-300 shadow-[0_0_0_4px_rgba(103,232,249,0.1)]" />
          </div>
          <p className="mt-3 text-sm font-semibold text-white">Local research mode</p>
          <p className="mt-1 text-xs leading-5 text-sidebar-foreground/55">Typed mock services are ready for API replacement.</p>
        </div>
      </div>
      <div className="border-t border-sidebar-border px-7 py-4 text-[10px] text-sidebar-foreground/40">OOC-V / 0.1.0-preview</div>
    </>
  );
}