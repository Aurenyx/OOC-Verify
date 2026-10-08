import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ErrorBoundary } from '@/components/error-boundary';
import { Toaster } from '@/components/ui/toaster';
import { TooltipProvider } from '@/components/ui/tooltip';
import { AppShell } from '@/components/app-shell';
import {
  ArrowRight,
  BookOpen,
  Check,
  ChevronDown,
  ChevronRight,
  CircleAlert,
  Copy,
  Database,
  Download,
  ExternalLink,
  FileSearch,
  Filter,
  Layers3,
  Link as LinkIcon,
  Loader2,
  LockKeyhole,
  Minus,
  Play,
  Plus,
  RotateCcw,
  Search,
  ShieldCheck,
  Sparkles,
  TerminalSquare,
  Trash2,
  Upload,
} from 'lucide-react';
import { Link, Route, Switch, Router as WouterRouter, useLocation, useRoute } from 'wouter';
import NotFound from '@/pages/not-found';
import {
  experimentConfigurations,
  mockService,
  pipelineStages,
  sampleImages,
  useVerifications,
  type DatasetSample,
  type Evidence,
  type EvidenceRelation,
  type ExperimentConfiguration,
  type Verification,
} from '@/services/mock-data';
import { apiService } from '@/services/api-service';

const queryClient = new QueryClient();

function PageIntro({ eyebrow, title, description, action }: { eyebrow: string; title: string; description: string; action?: ReactNode }) {
  return (
    <div className="animate-enter mb-8 flex flex-col justify-between gap-6 border-b border-border/80 pb-8 md:flex-row md:items-end">
      <div className="max-w-3xl">
        <div className="mb-3 flex items-center gap-2 font-mono text-[10px] font-medium uppercase tracking-[0.18em] text-primary">
          <span className="h-1.5 w-1.5 rounded-full bg-cyan-400" />
          {eyebrow}
        </div>
        <h1 className="text-balance text-3xl font-extrabold tracking-[-0.045em] text-foreground sm:text-4xl">{title}</h1>
        <p className="mt-3 max-w-2xl text-sm leading-6 text-muted-foreground">{description}</p>
      </div>
      {action}
    </div>
  );
}

function Overview() {
  const verifications = useVerifications();
  const consistentCount = verifications.filter((verification) => verification.prediction === 'Genuine').length;
  const misleadingCount = verifications.filter((verification) => verification.prediction === 'Misleading').length;
  const insufficientCount = verifications.filter((verification) => verification.evidence.some((item) => item.relation === 'insufficient')).length;

  return (
    <AppShell>
      <div className="relative overflow-hidden rounded-[28px] border border-indigo-100 bg-card px-6 py-8 shadow-[0_18px_70px_-34px_rgba(67,56,202,0.32)] sm:px-10 sm:py-11">
        <div className="soft-grid pointer-events-none absolute inset-0 opacity-70" />
        <div className="absolute -right-16 -top-20 h-64 w-64 rounded-full bg-violet-200/35 blur-3xl" />
        <div className="relative grid gap-10 lg:grid-cols-[1fr_310px] lg:items-center">
          <div className="max-w-2xl">
            <div className="mb-5 inline-flex items-center gap-2 rounded-full border border-indigo-100 bg-indigo-50/80 px-3 py-1.5 font-mono text-[10px] uppercase tracking-[0.17em] text-indigo-700">
              <Sparkles className="h-3.5 w-3.5" /> Explainable multimodal verification
            </div>
            <h1 className="text-balance text-4xl font-extrabold tracking-[-0.06em] text-slate-950 sm:text-6xl">
              OOC-<span className="bg-gradient-to-r from-indigo-600 via-violet-600 to-cyan-500 bg-clip-text text-transparent">Verify</span>
            </h1>
            <p className="mt-3 text-lg font-semibold text-slate-700">
              Explainable Out-of-Context Image-Caption Misinformation Detection
            </p>
            <p className="mt-3 max-w-xl text-base leading-7 text-slate-600">
              OOC-Verify helps researchers inspect how an image-caption pair moves from alignment signals to an evidence-backed decision.
            </p>
            <div className="mt-8 flex flex-wrap items-center gap-3">
              <Link href="/verify" data-testid="link-start-verification" className="focus-ring inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-indigo-600 to-violet-600 px-5 py-3 text-sm font-bold text-white shadow-lg shadow-indigo-500/20 transition-transform hover:-translate-y-0.5">
                Start a verification <ArrowRight className="h-4 w-4" />
              </Link>
              <Link href="/dataset" data-testid="link-browse-dataset" className="focus-ring inline-flex items-center gap-2 rounded-xl border border-border bg-white/75 px-5 py-3 text-sm font-semibold text-slate-700 hover:border-indigo-200 hover:text-indigo-700">
                Browse Dataset
              </Link>
            </div>
          </div>
          <PipelineOrb />
        </div>
      </div>

      <div className="mt-8 grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <MetricCard label="Total Verifications" value={String(verifications.length)} detail="Workspace records" icon={<FileSearch className="h-4 w-4" />} tone="violet" />
        <MetricCard label="Consistent" value={String(consistentCount)} detail="Genuine predictions" icon={<ShieldCheck className="h-4 w-4" />} tone="blue" />
        <MetricCard label="Potentially Misleading" value={String(misleadingCount)} detail="Needs contextual review" icon={<CircleAlert className="h-4 w-4" />} tone="cyan" />
        <MetricCard label="Insufficient Evidence" value={String(insufficientCount)} detail="Evidence state is explicit" icon={<BookOpen className="h-4 w-4" />} tone="slate" />
      </div>

      <div className="mt-10 grid gap-6 xl:grid-cols-[1.15fr_0.85fr]">
        <section className="rounded-2xl border border-border bg-card p-6 shadow-sm sm:p-7">
          <SectionHeading
            title="Recent verifications"
            eyebrow="Workspace activity"
            action={<Link href="/history" data-testid="link-see-all-history" className="focus-ring text-xs font-bold text-primary hover:underline">See all ({verifications.length})</Link>}
          />
          <div className="mt-5 divide-y divide-border">
            {verifications.slice(0, 5).map((verification) => (
              <VerificationRow key={verification.id} verification={verification} />
            ))}
          </div>
        </section>
        <section className="rounded-2xl border border-border bg-card p-6 shadow-sm sm:p-7">
          <SectionHeading title="How a decision is built" eyebrow="Inspectable by design" />
          <div className="mt-6 space-y-5">
            {[
              ['01', 'Align', 'Compare what the image shows with what the caption claims.'],
              ['02', 'Retrieve', 'Bring relevant, dated sources into the review surface.'],
              ['03', 'Explain', 'Keep the decision, reason, evidence, and explanation distinct.'],
            ].map(([number, title, copy]) => (
              <div key={number} className="flex gap-4" data-testid={`text-overview-principle-${number}`}>
                <span className="font-mono text-[11px] text-primary">{number}</span>
                <div>
                  <p className="text-sm font-bold text-foreground">{title}</p>
                  <p className="mt-1 text-xs leading-5 text-muted-foreground">{copy}</p>
                </div>
              </div>
            ))}
          </div>
          <div className="mt-7 rounded-xl border border-accent/80 bg-accent/30 p-4">
            <div className="flex items-center gap-2 text-xs font-bold text-accent-foreground">
              <LockKeyhole className="h-3.5 w-3.5" /> Local research mode active
            </div>
            <p className="mt-2 text-xs leading-5 text-accent-foreground/75">
              Multimodal verification runs locally on Antigravity localhost. Data and runs persist within your browser workspace.
            </p>
          </div>
        </section>
      </div>
    </AppShell>
  );
}

function PipelineOrb() {
  return (
    <div className="relative mx-auto flex h-[246px] w-[246px] items-center justify-center" aria-label="Seven stage verification pipeline" data-testid="visual-pipeline-orb">
      <div className="absolute inset-5 rounded-full border border-indigo-100 bg-gradient-to-br from-indigo-50 via-white to-cyan-50 shadow-inner" />
      <div className="absolute inset-12 rounded-full border border-dashed border-indigo-200 animate-[spin_60s_linear_infinite]" />
      <div className="relative flex h-24 w-24 flex-col items-center justify-center rounded-3xl bg-gradient-to-br from-indigo-600 to-violet-600 text-center text-white shadow-xl shadow-indigo-500/25">
        <ShieldCheck className="mb-1 h-7 w-7" />
        <span className="font-mono text-[9px] uppercase tracking-[0.15em]">OOC-V</span>
      </div>
      {['Input', 'Align', 'Reason', 'Evidence', 'Decide'].map((label, index) => {
        const positions = ['left-3 top-24', 'left-16 top-3', 'right-4 top-16', 'right-8 bottom-7', 'left-12 bottom-5'];
        return (
          <div
            key={label}
            className={`absolute ${positions[index]} rounded-full border border-white bg-white px-2.5 py-1 font-mono text-[9px] font-medium text-indigo-700 shadow-md`}
            data-testid={`text-orb-stage-${label.toLowerCase()}`}
          >
            {label}
          </div>
        );
      })}
    </div>
  );
}

function MetricCard({ label, value, detail, icon, tone }: { label: string; value: string; detail: string; icon: ReactNode; tone: 'violet' | 'blue' | 'cyan' | 'slate' }) {
  const tones = { violet: 'bg-violet-50 text-violet-700', blue: 'bg-blue-50 text-blue-700', cyan: 'bg-cyan-50 text-cyan-700', slate: 'bg-slate-100 text-slate-600' };
  return (
    <div className="rounded-2xl border border-border bg-card p-5 shadow-sm transition-all hover:shadow-md" data-testid={`metric-${label.toLowerCase().replaceAll(' ', '-')}`}>
      <div className={`flex h-9 w-9 items-center justify-center rounded-xl ${tones[tone]}`}>{icon}</div>
      <div className="mt-5 flex items-end justify-between gap-3">
        <div>
          <p className="text-xs font-semibold text-muted-foreground">{label}</p>
          <p className="mt-1 text-2xl font-extrabold tracking-[-0.04em] text-foreground">{value}</p>
        </div>
        <span className="text-right text-[10px] leading-4 text-muted-foreground">{detail}</span>
      </div>
    </div>
  );
}

function SectionHeading({ title, eyebrow, action }: { title: string; eyebrow: string; action?: ReactNode }) {
  return (
    <div className="flex items-end justify-between gap-4">
      <div>
        <p className="font-mono text-[10px] uppercase tracking-[0.16em] text-primary">{eyebrow}</p>
        <h2 className="mt-2 text-lg font-extrabold tracking-[-0.03em] text-foreground">{title}</h2>
      </div>
      {action}
    </div>
  );
}

function VerificationRow({ verification }: { verification: Verification }) {
  return (
    <Link href={`/results/${verification.id}`} data-testid={`link-recent-${verification.id}`} className="focus-ring group flex items-center gap-4 py-4">
      <img src={verification.image} alt="" className="h-12 w-12 rounded-xl object-cover ring-1 ring-black/5" />
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <p className="truncate text-sm font-bold text-foreground group-hover:text-primary">{verification.caption}</p>
          <StatusPill prediction={verification.prediction} />
        </div>
        <p className="mt-1 text-xs text-muted-foreground">
          {verification.dataset} <span className="mx-1 text-border">·</span> {verification.createdAt}
        </p>
      </div>
      <div className="hidden text-right sm:block">
        <p className="font-mono text-sm font-medium text-foreground">{Math.round(verification.confidenceScore * 100)}%</p>
        <p className="text-[10px] text-muted-foreground">Decision Confidence</p>
      </div>
      <ChevronRight className="h-4 w-4 shrink-0 text-muted-foreground transition-transform group-hover:translate-x-0.5" />
    </Link>
  );
}

function StatusPill({ prediction }: { prediction: Verification['prediction'] }) {
  return (
    <span
      className={`inline-flex rounded-full px-2 py-0.5 text-[10px] font-bold ${prediction === 'Genuine' ? 'bg-emerald-50 text-emerald-700 border border-emerald-200/50' : 'bg-rose-50 text-rose-700 border border-rose-200/50'
        }`}
      data-testid={`status-prediction-${prediction.toLowerCase()}`}
    >
      {prediction}
    </span>
  );
}

export function formatConfiguration(config?: string): string {
  if (!config) return 'Proposed OOC-Verify';
  const c = config.toLowerCase().trim();
  if (c === 'alignment_only' || c === 'alignment only') return 'Alignment Only';
  if (c === 'mllm_only' || c === 'mllm only') return 'MLLM Only';
  if (c === 'proposed' || c === 'proposed ooc-verify') return 'Proposed OOC-Verify';
  return config;
}

function VerifyPage() {
  const [caption, setCaption] = useState('Fire crews contain a western ridge blaze after overnight winds.');
  const [image, setImage] = useState(sampleImages.wildfire);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [dataset, setDataset] = useState('NewsCLIPpings');
  const [configuration, setConfiguration] = useState<'alignment_only' | 'mllm_only' | 'proposed'>('proposed');
  const [loadedSampleId, setLoadedSampleId] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [stageIndex, setStageIndex] = useState(-1);
  const [completed, setCompleted] = useState<Verification | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [showUrlInput, setShowUrlInput] = useState(false);
  const [customUrl, setCustomUrl] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);
  const stageTimerRef = useRef<number | null>(null);
  const [, setLocation] = useLocation();

  // Read URL query params on mount to support "Use in verification" from Dataset page
  useEffect(() => {
    const searchParams = new URLSearchParams(window.location.search);
    const sampleId = searchParams.get('sample') || searchParams.get('sampleId');
    if (sampleId) {
      const sample = mockService.getDatasetSample(sampleId);
      if (sample) {
        setCaption(sample.caption);
        setImage(sample.image);
        setSelectedFile(null);
        setDataset(sample.dataset);
        setLoadedSampleId(sample.id);
      }
    }
  }, []);

  useEffect(() => {
    return () => {
      if (stageTimerRef.current !== null) {
        window.clearInterval(stageTimerRef.current);
      }
    };
  }, []);

  const start = async () => {
    setCompleted(null);
    setErrorMessage(null);
    setStageIndex(0);
    setRunning(true);

    let stage = 0;
    stageTimerRef.current = window.setInterval(() => {
      stage = Math.min(stage + 1, pipelineStages.length - 1);
      setStageIndex(stage);
    }, 250);

    try {
      // Call the FastAPI verification endpoint via apiService
      const result = await apiService.verify({
        imageFile: selectedFile,
        imageUrl: image,
        caption,
        dataset,
        configuration,
      });

      if (stageTimerRef.current !== null) {
        window.clearInterval(stageTimerRef.current);
        stageTimerRef.current = null;
      }
      setStageIndex(pipelineStages.length);
      setCompleted(result);
    } catch (err: any) {
      if (stageTimerRef.current !== null) {
        window.clearInterval(stageTimerRef.current);
        stageTimerRef.current = null;
      }
      console.error('Verification error:', err);
      setErrorMessage(
        err?.message || 'Verification failed. Please check that the FastAPI backend is running.'
      );
    } finally {
      setRunning(false);
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      const reader = new FileReader();
      reader.onload = (event) => {
        if (event.target?.result) {
          setImage(event.target.result as string);
          setLoadedSampleId(null);
          setDataset('Custom Upload');
        }
      };
      reader.readAsDataURL(file);
    }
  };

  const handleApplyCustomUrl = () => {
    if (customUrl.trim()) {
      setImage(customUrl.trim());
      setSelectedFile(null);
      setLoadedSampleId(null);
      setDataset('Custom Image URL');
      setShowUrlInput(false);
      setCustomUrl('');
    }
  };

  const handleReset = () => {
    if (stageTimerRef.current !== null) {
      window.clearInterval(stageTimerRef.current);
      stageTimerRef.current = null;
    }
    setCaption('Fire crews contain a western ridge blaze after overnight winds.');
    setImage(sampleImages.wildfire);
    setSelectedFile(null);
    setDataset('NewsCLIPpings');
    setConfiguration('proposed');
    setLoadedSampleId(null);
    setCompleted(null);
    setErrorMessage(null);
    setStageIndex(-1);
    setRunning(false);
  };

  return (
    <AppShell>
      <PageIntro
        eyebrow="Verification workspace"
        title="Inspect an image-caption pair"
        description="Provide a paired input, then follow each stage as OOC-Verify builds an explainable report. Runs dynamically on Antigravity localhost."
        action={
          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={handleReset}
              className="focus-ring inline-flex items-center gap-1.5 rounded-xl border border-border bg-card px-3 py-2 font-mono text-[10px] uppercase tracking-[0.14em] text-muted-foreground hover:bg-muted hover:text-foreground"
            >
              <RotateCcw className="h-3 w-3" /> Reset inputs
            </button>
            <div className="hidden rounded-full border border-border bg-card px-3 py-2 font-mono text-[10px] uppercase tracking-[0.14em] text-muted-foreground md:block">
              Run ID <span className="text-foreground">{completed ? completed.id : running ? 'generating...' : 'pending'}</span>
            </div>
          </div>
        }
      />

      {loadedSampleId && (
        <div className="mb-6 flex items-center justify-between rounded-xl border border-indigo-100 bg-indigo-50/70 p-3.5 text-xs text-indigo-900 animate-enter">
          <div className="flex items-center gap-2">
            <Sparkles className="h-4 w-4 text-indigo-600" />
            <span>
              Loaded benchmark sample <strong className="font-mono text-indigo-700">{loadedSampleId}</strong> from <strong>{dataset}</strong>.
            </span>
          </div>
          <button
            type="button"
            onClick={() => setLoadedSampleId(null)}
            className="text-[11px] font-bold text-indigo-600 hover:text-indigo-800 hover:underline"
          >
            Dismiss badge
          </button>
        </div>
      )}

      <div className="grid gap-6 xl:grid-cols-[0.83fr_1.17fr]">
        <section className="rounded-2xl border border-border bg-card p-6 shadow-sm sm:p-7">
          <div className="flex items-center justify-between">
            <div>
              <p className="font-mono text-[10px] uppercase tracking-[0.16em] text-primary">01 / Input</p>
              <h2 className="mt-2 text-lg font-extrabold tracking-[-0.03em]">Image + caption</h2>
            </div>
            <span className="rounded-lg bg-muted px-2.5 py-1 font-mono text-[10px] text-muted-foreground">Required</span>
          </div>

          <div className="mt-6 overflow-hidden rounded-2xl border border-border bg-muted/40">
            <div className="relative aspect-[16/10]">
              <img src={image} alt="Selected verification input" className="h-full w-full object-cover" data-testid="img-verification-input" />
              <div className="absolute left-3 top-3 rounded-lg bg-slate-950/65 px-2.5 py-1.5 font-mono text-[10px] text-white backdrop-blur-sm">
                source / {dataset}
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-2 border-t border-border bg-white/80 p-3">
              <input type="file" ref={fileInputRef} onChange={handleFileUpload} accept="image/*" className="hidden" />
              <button
                type="button"
                data-testid="button-upload-image"
                onClick={() => fileInputRef.current?.click()}
                className="focus-ring inline-flex items-center gap-2 rounded-lg border border-border bg-white px-3 py-2 text-xs font-bold text-foreground hover:border-primary/40 hover:bg-slate-50"
              >
                <Upload className="h-3.5 w-3.5" /> Upload image
              </button>

              <button
                type="button"
                onClick={() => setShowUrlInput(!showUrlInput)}
                className="focus-ring inline-flex items-center gap-1.5 rounded-lg border border-border bg-white px-3 py-2 text-xs font-bold text-foreground hover:border-primary/40 hover:bg-slate-50"
              >
                <LinkIcon className="h-3.5 w-3.5" /> Image URL
              </button>

              {['NewsCLIPpings', 'VisualNews'].map((name) => (
                <button
                  key={name}
                  type="button"
                  data-testid={`button-select-dataset-${name.toLowerCase()}`}
                  onClick={() => {
                    setDataset(name);
                    setImage(name === 'VisualNews' ? sampleImages.parliament : sampleImages.wildfire);
                    setLoadedSampleId(null);
                  }}
                  className={`focus-ring rounded-lg px-3 py-2 text-xs font-bold ${dataset === name ? 'bg-indigo-50 text-indigo-700' : 'text-muted-foreground hover:bg-muted'
                    }`}
                >
                  {name} preset
                </button>
              ))}
            </div>

            {showUrlInput && (
              <div className="border-t border-border bg-slate-50 p-3 flex gap-2">
                <input
                  type="url"
                  placeholder="https://example.com/image.jpg"
                  value={customUrl}
                  onChange={(e) => setCustomUrl(e.target.value)}
                  className="focus-ring flex-1 rounded-lg border border-input bg-white px-3 py-1.5 text-xs outline-none"
                />
                <button
                  type="button"
                  onClick={handleApplyCustomUrl}
                  className="rounded-lg bg-indigo-600 px-3 py-1.5 text-xs font-bold text-white hover:bg-indigo-700"
                >
                  Apply
                </button>
              </div>
            )}
          </div>

          <label htmlFor="caption-input" className="mt-6 block text-xs font-bold text-foreground">
            Caption
          </label>
          <textarea
            id="caption-input"
            value={caption}
            disabled={running}
            onChange={(event) => setCaption(event.target.value)}
            data-testid="input-caption"
            className="focus-ring mt-2 min-h-[128px] w-full resize-y rounded-xl border border-input bg-white p-4 text-sm leading-6 text-foreground shadow-sm outline-none placeholder:text-muted-foreground focus:border-primary disabled:opacity-60"
            placeholder="Enter the claim associated with this image"
          />
          <div className="mt-2 flex items-center justify-between text-[10px] text-muted-foreground">
            <span>Use the claim wording you want to inspect.</span>
            <span data-testid="text-caption-count">{caption.length} characters</span>
          </div>

          <div className="mt-5">
            <div className="flex items-center justify-between">
              <label className="text-xs font-bold text-foreground">
                Verification Configuration
              </label>
              <span className="font-mono text-[10px] text-primary">
                {configuration === 'proposed'
                  ? 'Proposed OOC-Verify'
                  : configuration === 'alignment_only'
                  ? 'Alignment Only'
                  : 'MLLM Only'}
              </span>
            </div>
            <div className="mt-2 grid grid-cols-1 sm:grid-cols-3 gap-2">
              {[
                { id: 'alignment_only', label: 'Alignment Only' },
                { id: 'mllm_only', label: 'MLLM Only' },
                { id: 'proposed', label: 'Proposed OOC-Verify' },
              ].map((opt) => {
                const isSelected = configuration === opt.id;
                return (
                  <button
                    key={opt.id}
                    type="button"
                    disabled={running}
                    onClick={() => setConfiguration(opt.id as 'alignment_only' | 'mllm_only' | 'proposed')}
                    data-testid={`button-config-${opt.id}`}
                    className={`focus-ring relative flex items-center justify-center rounded-xl border px-3 py-2.5 text-xs font-bold transition-all ${
                      isSelected
                        ? 'border-indigo-600 bg-indigo-50/80 text-indigo-700 shadow-sm ring-1 ring-indigo-600/20'
                        : 'border-border bg-white text-muted-foreground hover:border-slate-300 hover:text-foreground'
                    } disabled:opacity-50`}
                  >
                    <span>{opt.label}</span>
                    {isSelected && (
                      <span className="ml-1.5 h-1.5 w-1.5 rounded-full bg-indigo-600" />
                    )}
                  </button>
                );
              })}
            </div>
          </div>

          <button
            type="button"
            disabled={running || !caption.trim()}
            onClick={start}
            data-testid="button-run-verification"
            className="focus-ring mt-6 flex w-full items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-indigo-600 to-violet-600 px-4 py-3.5 text-sm font-bold text-white shadow-lg shadow-indigo-500/15 transition-transform hover:-translate-y-0.5 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {running ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" /> Running pipeline
              </>
            ) : (
              <>
                <Play className="h-4 w-4" /> Run verification
              </>
            )}
          </button>

          {errorMessage && (
            <div className="mt-4 rounded-xl border border-rose-200 bg-rose-50 p-3.5 text-xs text-rose-800 animate-enter" data-testid="error-verification-message">
              <p className="font-bold">Backend Error</p>
              <p className="mt-0.5">{errorMessage}</p>
            </div>
          )}
        </section>

        <section className="rounded-2xl border border-border bg-card p-6 shadow-sm sm:p-7">
          <div className="flex items-center justify-between">
            <div>
              <p className="font-mono text-[10px] uppercase tracking-[0.16em] text-primary">02 / Pipeline</p>
              <h2 className="mt-2 text-lg font-extrabold tracking-[-0.03em]">Staged verification</h2>
            </div>
            <span
              className={`rounded-full px-2.5 py-1 text-[10px] font-bold ${completed ? 'bg-emerald-50 text-emerald-700' : running ? 'bg-indigo-50 text-indigo-700' : 'bg-muted text-muted-foreground'
                }`}
              data-testid="status-pipeline"
            >
              {completed ? 'Complete' : running ? 'In progress' : 'Ready'}
            </span>
          </div>

          <div className="mt-8 space-y-1">
            {pipelineStages.map((stage, index) => {
              const stageResult = completed?.pipelineStages?.find((s) => s.stage === stage.id);

              if (completed) {
                const status = (stageResult?.status || 'completed').toLowerCase();
                const isSkipped = status === 'skipped';
                const isFailed = status === 'failed';
                const isCompleted = status === 'completed';
                const isPending = status === 'pending';

                return (
                  <div
                    key={stage.id}
                    className={`relative flex gap-4 rounded-xl px-3 py-3.5 transition-colors ${
                      isSkipped
                        ? 'opacity-60 bg-muted/20'
                        : isFailed
                        ? 'bg-rose-50/50'
                        : 'bg-white'
                    }`}
                    data-testid={`pipeline-stage-${stage.id}`}
                  >
                    <div
                      className={`relative z-10 flex h-7 w-7 shrink-0 items-center justify-center rounded-full border text-[11px] font-bold ${
                        isCompleted
                          ? 'border-emerald-500 bg-emerald-600 text-white'
                          : isSkipped
                          ? 'border-slate-300 bg-slate-100 text-slate-400'
                          : isFailed
                          ? 'border-rose-500 bg-rose-600 text-white'
                          : 'border-border bg-white text-muted-foreground'
                      }`}
                    >
                      {isCompleted ? (
                        <Check className="h-3.5 w-3.5" />
                      ) : isSkipped ? (
                        <Minus className="h-3.5 w-3.5" />
                      ) : isFailed ? (
                        <CircleAlert className="h-3.5 w-3.5" />
                      ) : (
                        index + 1
                      )}
                    </div>
                    <div className="min-w-0 pt-0.5 flex-1">
                      <div className="flex items-center gap-2">
                        <p className={`text-sm font-bold ${isSkipped ? 'text-muted-foreground line-through decoration-slate-300' : isFailed ? 'text-rose-700' : 'text-foreground'}`}>
                          {stage.label}
                        </p>
                        {isSkipped && (
                          <span className="rounded-full bg-slate-100 px-2 py-0.5 text-[9px] font-bold text-slate-500 border border-slate-200">
                            Skipped
                          </span>
                        )}
                        {isFailed && (
                          <span className="rounded-full bg-rose-50 px-2 py-0.5 text-[9px] font-bold text-rose-700 border border-rose-200">
                            Failed
                          </span>
                        )}
                        {isPending && (
                          <span className="rounded-full bg-slate-50 px-2 py-0.5 text-[9px] font-bold text-slate-400 border border-slate-200">
                            Pending
                          </span>
                        )}
                        {isCompleted && (
                          <span className="rounded-full bg-emerald-50 px-2 py-0.5 text-[9px] font-bold text-emerald-700 border border-emerald-200">
                            Completed
                          </span>
                        )}
                      </div>
                      <p className="mt-0.5 text-xs text-muted-foreground">
                        {stageResult?.detail || stageDescription(stage.id)}
                      </p>
                    </div>
                    {index < pipelineStages.length - 1 && (
                      <span className={`absolute left-[26px] top-[42px] h-5 w-px ${isCompleted ? 'bg-emerald-300' : 'bg-border'}`} />
                    )}
                  </div>
                );
              }

              const active = running && index === stageIndex;
              const done = running && index < stageIndex;
              return (
                <div
                  key={stage.id}
                  className={`relative flex gap-4 rounded-xl px-3 py-3.5 transition-colors ${active ? 'bg-indigo-50/75' : ''}`}
                  data-testid={`pipeline-stage-${stage.id}`}
                >
                  <div
                    className={`relative z-10 flex h-7 w-7 shrink-0 items-center justify-center rounded-full border text-[11px] font-bold ${done
                        ? 'border-indigo-500 bg-indigo-600 text-white'
                        : active
                          ? 'border-indigo-400 bg-white text-indigo-700 shadow-sm'
                          : 'border-border bg-white text-muted-foreground'
                      }`}
                  >
                    {done ? <Check className="h-3.5 w-3.5" /> : index + 1}
                  </div>
                  <div className="min-w-0 pt-0.5">
                    <p className={`text-sm font-bold ${active ? 'text-indigo-800' : 'text-foreground'}`}>{stage.label}</p>
                    <p className="mt-0.5 text-xs text-muted-foreground">{stageDescription(stage.id)}</p>
                  </div>
                  {active && <Loader2 className="ml-auto mt-1 h-4 w-4 animate-spin text-indigo-600" />}
                  {index < pipelineStages.length - 1 && (
                    <span className={`absolute left-[26px] top-[42px] h-5 w-px ${done ? 'bg-indigo-300' : 'bg-border'}`} />
                  )}
                </div>
              );
            })}
          </div>

          {!running && !completed && (
            <div className="mt-6 rounded-xl border border-dashed border-border bg-muted/35 p-4 text-xs leading-5 text-muted-foreground" data-testid="status-pipeline-ready">
              Your explainable report will be assembled and persisted here after the seven-stage pipeline completes.
            </div>
          )}

          {completed && (
            <div className="mt-6 flex items-center justify-between rounded-xl border border-emerald-100 bg-emerald-50/60 p-4 animate-enter" data-testid="status-pipeline-complete">
              <div>
                <p className="text-xs font-bold text-emerald-800">
                  Report ready for review
                </p>
                <p className="mt-1 text-xs text-emerald-700/75">
                  Prediction: <strong>{completed.prediction}</strong> ({Math.round(completed.confidenceScore * 100)}% confidence) · Configuration: <strong>{formatConfiguration(completed.configuration)}</strong>.
                </p>
              </div>
              <button
                type="button"
                data-testid="button-open-completed-report"
                onClick={() => setLocation(`/results/${completed.id}`)}
                className="focus-ring rounded-lg bg-emerald-700 px-3.5 py-2 text-xs font-bold text-white shadow-sm hover:bg-emerald-800"
              >
                Open report
              </button>
            </div>
          )}
        </section>
      </div>
    </AppShell>
  );
}

function stageDescription(stage: string) {
  const descriptions: Record<string, string> = {
    input_validation: 'Validate image dimensions, formats, and caption syntax.',
    image_text_analysis: 'Extract observable visual entities and linguistic assertions.',
    cross_modal_alignment: 'Evaluate semantic coherence between image and claim.',
    mllm_reasoning: 'Compose a multimodal contextual reasoning chain.',
    evidence_retrieval: 'Query verified archival and contemporary news reports.',
    evidence_analysis: 'Assess source relevance, stance, and corroboration metrics.',
    final_decision: 'Synthesize the structured verification report and confidence.',
  };
  return descriptions[stage] ?? '';
}

function ResultsPage() {
  const [, params] = useRoute('/results/:id');
  const verifications = useVerifications();
  const verification = mockService.getVerification(params?.id ?? '') ?? verifications[0];
  return (
    <AppShell>
      <ReportPage verification={verification} />
    </AppShell>
  );
}

function ReportPage({ verification }: { verification: Verification }) {
  const [expanded, setExpanded] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  const handleCopyId = () => {
    navigator.clipboard.writeText(verification.id);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleExportJson = () => {
    const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(verification, null, 2));
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute('href', dataStr);
    downloadAnchor.setAttribute('download', `${verification.id}-verification-report.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  };

  return (
    <>
      <PageIntro
        eyebrow="Structured report"
        title="Verification report"
        description="A review surface that keeps the prediction, reasoning, evidence, and explanation separate."
        action={
          <div className="flex flex-wrap items-center gap-2">
            <Link
              href="/verify"
              data-testid="link-new-verification"
              className="focus-ring inline-flex items-center gap-2 rounded-xl border border-border bg-card px-4 py-2.5 text-xs font-bold text-foreground hover:border-primary/40 hover:bg-slate-50"
            >
              <Plus className="h-3.5 w-3.5" /> New verification
            </Link>
            <button
              type="button"
              onClick={handleExportJson}
              className="focus-ring inline-flex items-center gap-2 rounded-xl border border-border bg-card px-4 py-2.5 text-xs font-bold text-foreground hover:border-primary/40 hover:bg-slate-50"
            >
              <Download className="h-3.5 w-3.5" /> Export JSON
            </button>
            <button
              type="button"
              data-testid="button-copy-report-id"
              onClick={handleCopyId}
              className="focus-ring inline-flex items-center gap-2 rounded-xl bg-slate-900 px-4 py-2.5 text-xs font-bold text-white hover:bg-slate-800 transition-colors"
            >
              {copied ? <Check className="h-3.5 w-3.5 text-emerald-400" /> : <Copy className="h-3.5 w-3.5" />}
              {copied ? 'Copied ID!' : 'Copy ID'}
            </button>
          </div>
        }
      />
      <div className="mb-6 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
        <span className="font-mono text-[10px] text-primary bg-indigo-50 px-2 py-0.5 rounded border border-indigo-100">{verification.id}</span>
        <span>·</span>
        <span className="rounded-full bg-violet-50 px-2.5 py-0.5 text-[11px] font-bold text-violet-700 border border-violet-200" data-testid="report-configuration">
          {formatConfiguration(verification.configuration)}
        </span>
        <span>·</span>
        <span>{verification.dataset}</span>
        <span>·</span>
        <span>{verification.createdAt}</span>
      </div>
      <div className="grid gap-6 xl:grid-cols-[1.08fr_0.92fr]">
        <div className="space-y-6">
          <section className="overflow-hidden rounded-2xl border border-border bg-card shadow-sm">
            <div className="grid md:grid-cols-[0.9fr_1.1fr]">
              <img src={verification.image} alt="Verification input" className="h-full min-h-[260px] w-full object-cover" data-testid="img-report-input" />
              <div className="p-6 sm:p-7">
                <div className="flex items-center justify-between">
                  <p className="font-mono text-[10px] uppercase tracking-[0.16em] text-primary">Input pair</p>
                  <StatusPill prediction={verification.prediction} />
                </div>
                <p className="mt-5 text-lg font-bold leading-7 tracking-[-0.02em] text-foreground" data-testid="text-report-caption">
                  “{verification.caption}”
                </p>
                <div className="mt-6 grid grid-cols-2 gap-3">
                  <InfoDatum label="Configuration" value={formatConfiguration(verification.configuration)} />
                  <InfoDatum label="Dataset" value={verification.dataset} />
                  <InfoDatum label="Ground truth label" value={verification.groundTruthLabel || 'None (Unlabeled / Integration test)'} />
                  <InfoDatum
                    label="CLIP Alignment Score"
                    value={
                      typeof verification.clipScore === 'number' && !Number.isNaN(verification.clipScore)
                        ? verification.clipScore.toFixed(4)
                        : typeof verification.alignmentScore === 'number' && !Number.isNaN(verification.alignmentScore)
                        ? verification.alignmentScore.toFixed(4)
                        : 'Skipped'
                    }
                  />
                </div>
              </div>
            </div>
          </section>
          <section className="rounded-2xl border border-border bg-card p-6 shadow-sm sm:p-7">
            <SectionHeading
              title="Evidence"
              eyebrow="03 / Retrieved context"
              action={<span className="font-mono text-[10px] text-muted-foreground">{verification.evidence.length} sources retrieved</span>}
            />
            <div className="mt-5 space-y-3">
              {verification.evidence.map((item, index) => (
                <EvidenceCard
                  key={`${item.source}-${index}`}
                  evidence={item}
                  index={index}
                  expanded={expanded === `${item.source}-${index}`}
                  onToggle={() => setExpanded(expanded === `${item.source}-${index}` ? null : `${item.source}-${index}`)}
                />
              ))}
            </div>
          </section>
        </div>
        <div className="space-y-6">
          <section className="rounded-2xl border border-indigo-100 bg-gradient-to-br from-indigo-50 via-white to-cyan-50 p-6 shadow-sm sm:p-7" data-testid="card-prediction">
            <div className="flex items-start justify-between">
              <div>
                <p className="font-mono text-[10px] uppercase tracking-[0.16em] text-indigo-700">01 / Prediction</p>
                <p
                  className={`mt-4 text-4xl font-extrabold tracking-[-0.06em] ${verification.prediction === 'Genuine' ? 'text-emerald-700' : 'text-rose-700'
                    }`}
                  data-testid="text-prediction"
                >
                  {verification.prediction}
                </p>
              </div>
              <div className="rounded-2xl bg-white/85 p-3 text-indigo-600 shadow-sm">
                <ShieldCheck className="h-6 w-6" />
              </div>
            </div>
            <div className="mt-8 border-t border-indigo-100 pt-5">
              <div className="flex items-end justify-between">
                <div>
                  <p className="text-xs font-bold text-slate-600">Decision Confidence</p>
                  <p className="mt-1 text-xs text-slate-500">Cross-modal agreement score.</p>
                </div>
                <span className="font-mono text-2xl font-medium text-slate-900" data-testid="text-decision-confidence">
                  {typeof verification.confidenceScore === 'number' && !Number.isNaN(verification.confidenceScore)
                    ? `${Math.round(verification.confidenceScore * 100)}%`
                    : '0%'}
                </span>
              </div>
              <div className="mt-4 h-2 overflow-hidden rounded-full bg-indigo-100">
                <div
                  className="h-full rounded-full bg-gradient-to-r from-indigo-500 to-cyan-400 transition-all duration-1000"
                  style={{ width: `${Math.max(0, Math.min(100, (verification.confidenceScore ?? 0) * 100))}%` }}
                />
              </div>
            </div>
          </section>
          <ReportBlock number="02" title="Reason" icon={<CircleAlert className="h-4 w-4" />}>
            <p data-testid="text-report-reason">{verification.reason}</p>
            <div className="mt-4 inline-flex items-center gap-2 rounded-lg bg-muted px-3 py-2 text-xs font-semibold text-muted-foreground">
              <span className="h-1.5 w-1.5 rounded-full bg-violet-500" /> {verification.inconsistencyType}
            </div>
          </ReportBlock>
          <ReportBlock number="04" title="Explanation" icon={<Sparkles className="h-4 w-4" />}>
            <p data-testid="text-report-explanation">{verification.explanation}</p>
          </ReportBlock>
          <section className="rounded-2xl border border-border bg-card p-6 shadow-sm sm:p-7">
            <SectionHeading title="Pipeline trace" eyebrow="Run metadata" />
            <div className="mt-5 grid grid-cols-2 gap-3">
              {pipelineStages.map((stage, index) => {
                const stageResult = verification.pipelineStages?.find(
                  (s) => s.stage === stage.id || s.stage.toLowerCase() === stage.id.toLowerCase()
                );
                const status = (stageResult?.status || 'completed').toLowerCase();

                let statusLabel = 'Completed';
                let statusColor = 'text-emerald-600';
                let icon = <Check className="h-2.5 w-2.5" />;
                let cardBg = 'bg-muted/30';

                if (status === 'skipped') {
                  statusLabel = 'Skipped';
                  statusColor = 'text-slate-400';
                  icon = <Minus className="h-2.5 w-2.5" />;
                  cardBg = 'bg-muted/15 opacity-60';
                } else if (status === 'failed') {
                  statusLabel = 'Failed';
                  statusColor = 'text-rose-600';
                  icon = <CircleAlert className="h-2.5 w-2.5" />;
                  cardBg = 'bg-rose-50/30 border-rose-200/50';
                }

                return (
                  <div className={`rounded-xl border border-border p-3 transition-colors ${cardBg}`} key={stage.id} data-testid={`trace-stage-${stage.id}`}>
                    <div className="flex items-center justify-between">
                      <span className="font-mono text-[10px] text-primary">0{index + 1}</span>
                      {typeof stageResult?.score === 'number' && !Number.isNaN(stageResult.score) && (
                        <span className="font-mono text-[9px] text-muted-foreground">score: {stageResult.score.toFixed(2)}</span>
                      )}
                    </div>
                    <p className="mt-2 text-xs font-bold text-foreground">{stage.short}</p>
                    <p className={`mt-1 text-[10px] font-semibold flex items-center gap-1 ${statusColor}`}>
                      {icon} {statusLabel}
                    </p>
                    {stageResult?.detail && (
                      <p className="mt-1 line-clamp-1 text-[9px] text-muted-foreground" title={stageResult.detail}>
                        {stageResult.detail}
                      </p>
                    )}
                  </div>
                );
              })}
            </div>
          </section>
        </div>
      </div>
    </>
  );
}

function InfoDatum({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border border-border bg-muted/35 p-3">
      <p className="text-[10px] text-muted-foreground">{label}</p>
      <p className="mt-1 text-xs font-bold text-foreground">{value}</p>
    </div>
  );
}

function ReportBlock({ number, title, icon, children }: { number: string; title: string; icon: ReactNode; children: ReactNode }) {
  return (
    <section className="rounded-2xl border border-border bg-card p-6 shadow-sm sm:p-7">
      <div className="flex items-center gap-3">
        <span className="font-mono text-[10px] text-primary">{number}</span>
        <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-violet-50 text-violet-700">{icon}</span>
        <h2 className="text-lg font-extrabold tracking-[-0.03em]">{title}</h2>
      </div>
      <div className="mt-5 text-sm leading-7 text-muted-foreground">{children}</div>
    </section>
  );
}

function EvidenceCard({ evidence, index, expanded, onToggle }: { evidence: Evidence; index: number; expanded: boolean; onToggle: () => void }) {
  const relationStyle: Record<EvidenceRelation, string> = {
    supports: 'bg-emerald-50 text-emerald-700 border border-emerald-200/50',
    contradicts: 'bg-rose-50 text-rose-700 border border-rose-200/50',
    insufficient: 'bg-amber-50 text-amber-700 border border-amber-200/50',
    irrelevant: 'bg-slate-100 text-slate-600 border border-slate-200/50',
  };
  return (
    <article className="rounded-xl border border-border bg-white p-4 transition-shadow hover:shadow-md" data-testid={`card-evidence-${index}`}>
      <div className="flex items-start gap-3">
        <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-slate-900 text-white">
          <LinkIcon className="h-4 w-4" />
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <span className="font-mono text-[10px] font-medium uppercase tracking-[0.12em] text-primary">{evidence.source}</span>
            <span className={`rounded-full px-2 py-0.5 text-[10px] font-bold ${relationStyle[evidence.relation]}`} data-testid={`status-evidence-relation-${index}`}>
              {evidence.relation}
            </span>
          </div>
          <h3 className="mt-2 text-sm font-bold leading-5 text-foreground">{evidence.title}</h3>
          <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1 text-[10px] text-muted-foreground">
            <span>Published: {evidence.publishedDate}</span>
            <span>Retrieved: {evidence.retrievedDate}</span>
            <span className="font-mono text-foreground font-semibold">Relevance: {Math.round(evidence.relevance * 100)}%</span>
          </div>
        </div>
        <button
          type="button"
          aria-label={expanded ? 'Collapse evidence' : 'Expand evidence'}
          data-testid={`button-toggle-evidence-${index}`}
          onClick={onToggle}
          className="focus-ring rounded-lg p-1.5 text-muted-foreground hover:bg-muted hover:text-foreground"
        >
          {expanded ? <Minus className="h-4 w-4" /> : <Plus className="h-4 w-4" />}
        </button>
      </div>
      {expanded && (
        <div className="mt-4 border-t border-border pt-4 animate-enter">
          <p className="text-xs leading-5 text-muted-foreground">{evidence.excerpt}</p>
          <a
            href={evidence.url}
            target="_blank"
            rel="noreferrer"
            data-testid={`link-evidence-source-${index}`}
            className="focus-ring mt-3 inline-flex items-center gap-1.5 text-xs font-bold text-primary hover:underline"
          >
            Open external source <ExternalLink className="h-3 w-3" />
          </a>
        </div>
      )}
    </article>
  );
}

function HistoryPage() {
  const [query, setQuery] = useState('');
  const [dataset, setDataset] = useState('All datasets');
  const [prediction, setPrediction] = useState('All predictions');
  const verifications = useVerifications();

  const rows = useMemo(
    () =>
      verifications.filter(
        (item) =>
          (dataset === 'All datasets' || item.dataset === dataset) &&
          (prediction === 'All predictions' || item.prediction === prediction) &&
          item.caption.toLowerCase().includes(query.toLowerCase())
      ),
    [dataset, prediction, query, verifications]
  );

  return (
    <AppShell>
      <PageIntro
        eyebrow="Verification archive"
        title="History"
        description="Browse persisted verification runs and open any structured report for closer review."
        action={
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => mockService.resetVerifications()}
              className="focus-ring inline-flex items-center gap-1.5 rounded-xl border border-border bg-card px-3.5 py-2.5 text-xs font-bold text-muted-foreground hover:bg-muted hover:text-foreground"
            >
              <RotateCcw className="h-3.5 w-3.5" /> Reset archive
            </button>
            <Link
              href="/verify"
              data-testid="link-history-new"
              className="focus-ring inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-indigo-600 to-violet-600 px-4 py-2.5 text-xs font-bold text-white shadow-lg shadow-indigo-500/15"
            >
              <Plus className="h-3.5 w-3.5" /> New verification
            </Link>
          </div>
        }
      />
      <section className="rounded-2xl border border-border bg-card shadow-sm">
        <div className="flex flex-col gap-3 border-b border-border p-4 sm:flex-row sm:items-center sm:p-5">
          <div className="relative min-w-0 flex-1">
            <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            <input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              data-testid="input-history-search"
              className="focus-ring h-10 w-full rounded-xl border border-input bg-white pl-9 pr-3 text-sm outline-none focus:border-primary"
              placeholder="Search captions"
            />
          </div>
          <Filter className="hidden h-4 w-4 text-muted-foreground sm:block" />
          <select
            value={dataset}
            onChange={(event) => setDataset(event.target.value)}
            data-testid="select-history-dataset"
            className="focus-ring h-10 rounded-xl border border-input bg-white px-3 text-xs font-semibold outline-none"
          >
            <option>All datasets</option>
            <option>NewsCLIPpings</option>
            <option>VisualNews</option>
            <option>Custom Upload</option>
          </select>
          <select
            value={prediction}
            onChange={(event) => setPrediction(event.target.value)}
            data-testid="select-history-prediction"
            className="focus-ring h-10 rounded-xl border border-input bg-white px-3 text-xs font-semibold outline-none"
          >
            <option>All predictions</option>
            <option>Genuine</option>
            <option>Misleading</option>
          </select>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[680px] text-left">
            <thead className="bg-muted/45 text-[10px] uppercase tracking-[0.13em] text-muted-foreground">
              <tr>
                <th className="px-5 py-3 font-semibold">Input</th>
                <th className="px-5 py-3 font-semibold">Prediction</th>
                <th className="px-5 py-3 font-semibold">Decision Confidence</th>
                <th className="px-5 py-3 font-semibold">Dataset</th>
                <th className="px-5 py-3 font-semibold">Created</th>
                <th className="px-5 py-3" />
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {rows.map((item) => (
                <tr key={item.id} className="group hover:bg-muted/25 transition-colors" data-testid={`row-history-${item.id}`}>
                  <td className="max-w-[280px] px-5 py-4">
                    <div className="flex items-center gap-3">
                      <img src={item.image} alt="" className="h-9 w-9 rounded-lg object-cover ring-1 ring-black/5" />
                      <span className="truncate text-sm font-semibold text-foreground">{item.caption}</span>
                    </div>
                  </td>
                  <td className="px-5 py-4">
                    <StatusPill prediction={item.prediction} />
                  </td>
                  <td className="px-5 py-4 font-mono text-xs text-foreground">{Math.round(item.confidenceScore * 100)}%</td>
                  <td className="px-5 py-4 text-xs text-muted-foreground">{item.dataset}</td>
                  <td className="px-5 py-4 text-xs text-muted-foreground">{item.createdAt}</td>
                  <td className="px-5 py-4 text-right">
                    <div className="flex items-center justify-end gap-2">
                      <Link
                        href={`/results/${item.id}`}
                        data-testid={`link-history-report-${item.id}`}
                        className="focus-ring inline-flex items-center gap-1 text-xs font-bold text-primary hover:underline"
                      >
                        Report <ArrowRight className="h-3 w-3" />
                      </Link>
                      <button
                        type="button"
                        onClick={() => mockService.deleteVerification(item.id)}
                        title="Delete from history"
                        className="p-1 text-muted-foreground hover:text-rose-600 rounded"
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {rows.length === 0 && (
          <div className="p-12 text-center" data-testid="status-history-empty">
            <Search className="mx-auto h-6 w-6 text-muted-foreground" />
            <p className="mt-3 text-sm font-bold text-foreground">No matching runs</p>
            <p className="mt-1 text-xs text-muted-foreground">Try another filter or search phrase.</p>
          </div>
        )}
        <div className="border-t border-border px-5 py-3 text-xs text-muted-foreground">
          {rows.length} of {verifications.length} records shown
        </div>
      </section>
    </AppShell>
  );
}

function ExperimentsPage() {
  const [selected, setSelected] = useState<string | null>(null);
  return (
    <AppShell>
      <PageIntro
        eyebrow="Research design space"
        title="Experiment configurations"
        description="Compare the intended stages and outputs of each configuration without implying benchmark scores or research claims."
        action={<div className="rounded-full border border-border bg-card px-3 py-2 font-mono text-[10px] uppercase tracking-[0.14em] text-muted-foreground">3 configurations</div>}
      />
      <div className="grid gap-5 xl:grid-cols-3">
        {experimentConfigurations.map((config, index) => (
          <ExperimentCard
            key={config.name}
            config={config}
            index={index}
            selected={selected === config.name}
            onSelect={() => setSelected(selected === config.name ? null : config.name)}
          />
        ))}
      </div>
      <section className="mt-8 rounded-2xl border border-border bg-card p-6 shadow-sm sm:p-7">
        <SectionHeading title="Configuration lens" eyebrow="Read the comparison" />
        <div className="mt-5 grid gap-4 md:grid-cols-3">
          {[
            ['Stages', 'What the system is asked to do'],
            ['Outputs', 'What the researcher can inspect'],
            ['Scores', 'Not included in this preview'],
          ].map(([title, copy], index) => (
            <div key={title} className="rounded-xl border border-border bg-muted/30 p-4" data-testid={`text-experiment-lens-${index}`}>
              <p className="text-sm font-bold">{title}</p>
              <p className="mt-2 text-xs leading-5 text-muted-foreground">{copy}</p>
            </div>
          ))}
        </div>
      </section>
    </AppShell>
  );
}

function ExperimentCard({
  config,
  index,
  selected,
  onSelect,
}: {
  config: ExperimentConfiguration;
  index: number;
  selected: boolean;
  onSelect: () => void;
}) {
  const tone: Record<string, string> = {
    violet: 'from-violet-500 to-indigo-500',
    blue: 'from-blue-500 to-indigo-500',
    cyan: 'from-cyan-500 to-blue-500',
  };
  return (
    <article
      className={`rounded-2xl border bg-card p-6 shadow-sm transition-shadow hover:shadow-md ${selected ? 'border-primary ring-2 ring-primary/10' : 'border-border'
        }`}
      data-testid={`card-experiment-${index}`}
    >
      <div className="flex items-start justify-between gap-4">
        <div className={`flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br ${tone[config.tone]} text-white`}>
          <TerminalSquare className="h-5 w-5" />
        </div>
        <span className="rounded-full bg-muted px-2.5 py-1 font-mono text-[9px] uppercase tracking-[0.14em] text-muted-foreground">{config.type}</span>
      </div>
      <h2 className="mt-6 text-xl font-extrabold tracking-[-0.04em]">{config.name}</h2>
      <p className="mt-3 min-h-[60px] text-sm leading-6 text-muted-foreground">{config.description}</p>
      <div className="mt-6 border-t border-border pt-5">
        <p className="font-mono text-[10px] uppercase tracking-[0.15em] text-primary">Stages / {config.stages.length}</p>
        <div className="mt-3 flex flex-wrap gap-2">
          {config.stages.map((stage, stageIndex) => (
            <span key={stage} className="inline-flex items-center gap-1.5 rounded-lg border border-border bg-muted/35 px-2.5 py-1.5 text-[10px] font-semibold text-foreground">
              <span className="font-mono text-primary">{String(stageIndex + 1).padStart(2, '0')}</span>
              {stage}
            </span>
          ))}
        </div>
      </div>
      <div className="mt-5 border-t border-border pt-5">
        <p className="font-mono text-[10px] uppercase tracking-[0.15em] text-primary">Outputs</p>
        <ul className="mt-3 space-y-2">
          {config.outputs.map((output) => (
            <li key={output} className="flex items-center gap-2 text-xs font-semibold text-foreground">
              <Check className="h-3.5 w-3.5 text-cyan-600" /> {output}
            </li>
          ))}
        </ul>
      </div>
      <div className="mt-6 flex flex-col gap-2">
        <button
          type="button"
          onClick={onSelect}
          data-testid={`button-inspect-experiment-${index}`}
          className="focus-ring flex w-full items-center justify-center gap-2 rounded-xl border border-border py-2.5 text-xs font-bold text-foreground hover:border-primary/40 hover:text-primary"
        >
          {selected ? 'Hide configuration note' : 'Inspect configuration'} <ChevronDown className={`h-3.5 w-3.5 transition-transform ${selected ? 'rotate-180' : ''}`} />
        </button>
        <Link
          href="/verify"
          className="focus-ring flex w-full items-center justify-center gap-2 rounded-xl bg-slate-100 py-2.5 text-xs font-bold text-slate-700 hover:bg-slate-200"
        >
          Test in verification workspace <ArrowRight className="h-3.5 w-3.5" />
        </Link>
      </div>
      {selected && (
        <div className="mt-3 rounded-xl bg-indigo-50 p-3 text-xs leading-5 text-indigo-800 animate-enter" data-testid={`text-experiment-note-${index}`}>
          This architecture specifies the multimodal evaluation strategy. Antigravity local mode executes this structured review pipeline.
        </div>
      )}
    </article>
  );
}

function DatasetPage() {
  const [search, setSearch] = useState('');
  const [source, setSource] = useState('All datasets');
  const [split, setSplit] = useState('All splits');
  const samples = useMemo(
    () =>
      mockService
        .listDatasetSamples()
        .filter(
          (sample) =>
            (source === 'All datasets' || sample.dataset === source) &&
            (split === 'All splits' || sample.split === split) &&
            `${sample.caption} ${sample.id}`.toLowerCase().includes(search.toLowerCase())
        ),
    [search, source, split]
  );

  return (
    <AppShell>
      <PageIntro
        eyebrow="Benchmark sample browser"
        title="Dataset browser"
        description="Inspect representative sample records from NewsCLIPpings and VisualNews before choosing an input for verification."
        action={
          <div className="flex items-center gap-2 rounded-full border border-border bg-card px-3 py-2 font-mono text-[10px] uppercase tracking-[0.14em] text-muted-foreground">
            <Database className="h-4 w-4" /> {mockService.listDatasetSamples().length} samples available
          </div>
        }
      />
      <section className="rounded-2xl border border-border bg-card p-4 shadow-sm sm:p-5">
        <div className="flex flex-col gap-3 lg:flex-row">
          <div className="relative flex-1">
            <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            <input
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              data-testid="input-dataset-search"
              className="focus-ring h-10 w-full rounded-xl border border-input bg-white pl-9 pr-3 text-sm outline-none focus:border-primary"
              placeholder="Search sample IDs or captions"
            />
          </div>
          <select
            value={source}
            onChange={(event) => setSource(event.target.value)}
            data-testid="select-dataset-source"
            className="focus-ring h-10 rounded-xl border border-input bg-white px-3 text-xs font-semibold outline-none"
          >
            <option>All datasets</option>
            <option>NewsCLIPpings</option>
            <option>VisualNews</option>
          </select>
          <select
            value={split}
            onChange={(event) => setSplit(event.target.value)}
            data-testid="select-dataset-split"
            className="focus-ring h-10 rounded-xl border border-input bg-white px-3 text-xs font-semibold outline-none"
          >
            <option>All splits</option>
            <option>train</option>
            <option>validation</option>
            <option>test</option>
          </select>
        </div>
        <div className="mt-6 grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          {samples.map((sample) => (
            <DatasetCard key={sample.id} sample={sample} />
          ))}
        </div>
        {samples.length === 0 && (
          <div className="p-12 text-center" data-testid="status-dataset-empty">
            <p className="text-sm font-bold">No samples match those filters.</p>
          </div>
        )}
      </section>
    </AppShell>
  );
}

function DatasetCard({ sample }: { sample: DatasetSample }) {
  const [, setLocation] = useLocation();

  const handleUseSample = () => {
    // Navigate with sample query parameter so Verify page immediately populates it
    setLocation(`/verify?sample=${encodeURIComponent(sample.id)}`);
  };

  return (
    <article className="group overflow-hidden rounded-2xl border border-border bg-white shadow-sm transition-shadow hover:shadow-md" data-testid={`card-dataset-sample-${sample.id}`}>
      <div className="relative aspect-[16/9] overflow-hidden bg-muted">
        <img src={sample.image} alt="" className="h-full w-full object-cover transition-transform duration-500 group-hover:scale-105" />
        <div className="absolute left-3 top-3 rounded-full bg-slate-950/65 px-2.5 py-1 font-mono text-[9px] uppercase tracking-[0.12em] text-white backdrop-blur-sm">
          {sample.dataset}
        </div>
        <div
          className={`absolute right-3 top-3 rounded-full px-2.5 py-1 text-[9px] font-bold ${sample.label === 'Genuine' ? 'bg-emerald-100 text-emerald-800 border border-emerald-200' : 'bg-rose-100 text-rose-800 border border-rose-200'
            }`}
        >
          {sample.label}
        </div>
      </div>
      <div className="p-4">
        <div className="flex items-center justify-between">
          <span className="font-mono text-[10px] text-primary bg-indigo-50 px-2 py-0.5 rounded">{sample.id}</span>
          <span className="text-[10px] capitalize text-muted-foreground">{sample.split} split</span>
        </div>
        <p className="mt-3 line-clamp-2 text-sm font-semibold leading-5 text-foreground">{sample.caption}</p>
        <button
          type="button"
          onClick={handleUseSample}
          data-testid={`button-use-sample-${sample.id}`}
          className="focus-ring mt-4 flex w-full items-center justify-center gap-2 rounded-lg border border-border py-2 text-xs font-bold text-foreground hover:border-primary/40 hover:text-primary transition-colors"
        >
          Use in verification <ArrowRight className="h-3.5 w-3.5" />
        </button>
      </div>
    </article>
  );
}

function Router() {
  return (
    <RoutedErrorBoundary>
      <Switch>
        <Route path="/" component={Overview} />
        <Route path="/verify" component={VerifyPage} />
        <Route path="/results/:id" component={ResultsPage} />
        <Route path="/history" component={HistoryPage} />
        <Route path="/experiments" component={ExperimentsPage} />
        <Route path="/dataset" component={DatasetPage} />
        <Route component={NotFound} />
      </Switch>
    </RoutedErrorBoundary>
  );
}

function RoutedErrorBoundary({ children }: { children: ReactNode }) {
  const [location] = useLocation();
  return <ErrorBoundary resetKey={location}>{children}</ErrorBoundary>;
}

function App() {
  useEffect(() => {
    document.title = 'OOC-Verify — Explainable Out-of-Context Misinformation Detection';
  }, []);

  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <WouterRouter base={import.meta.env.BASE_URL.replace(/\/$/, '')}>
          <Router />
        </WouterRouter>
        <Toaster />
      </TooltipProvider>
    </QueryClientProvider>
  );
}

export default App;