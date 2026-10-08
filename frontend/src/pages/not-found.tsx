import { AppShell } from '@/components/app-shell';
import { AlertCircle, ArrowLeft, Home } from 'lucide-react';
import { Link } from 'wouter';

export default function NotFound() {
  return (
    <AppShell>
      <div className="flex min-h-[60vh] flex-col items-center justify-center text-center">
        <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-rose-50 text-rose-600 shadow-sm border border-rose-100">
          <AlertCircle className="h-8 w-8" />
        </div>
        <h1 className="mt-6 text-3xl font-extrabold tracking-[-0.04em] text-foreground sm:text-4xl">
          Page Not Found
        </h1>
        <p className="mt-3 max-w-md text-sm leading-6 text-muted-foreground">
          The requested research view or report identifier does not exist or may have been relocated.
        </p>
        <div className="mt-8 flex flex-wrap items-center justify-center gap-3">
          <Link
            href="/"
            className="focus-ring inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-indigo-600 to-violet-600 px-5 py-2.5 text-xs font-bold text-white shadow-md shadow-indigo-500/20"
          >
            <Home className="h-4 w-4" /> Return to Overview
          </Link>
          <Link
            href="/history"
            className="focus-ring inline-flex items-center gap-2 rounded-xl border border-border bg-card px-5 py-2.5 text-xs font-bold text-foreground hover:bg-muted"
          >
            <ArrowLeft className="h-4 w-4" /> View History
          </Link>
        </div>
      </div>
    </AppShell>
  );
}
