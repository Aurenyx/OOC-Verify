import { useEffect, useState } from 'react';

export type Prediction = 'Genuine' | 'Misleading';
export type EvidenceRelation = 'supports' | 'contradicts' | 'insufficient' | 'irrelevant';
export type PipelineStage =
  | 'input_validation'
  | 'image_text_analysis'
  | 'cross_modal_alignment'
  | 'mllm_reasoning'
  | 'evidence_retrieval'
  | 'evidence_analysis'
  | 'final_decision';

export type Evidence = {
  source: string;
  title: string;
  publishedDate: string;
  retrievedDate: string;
  relation: EvidenceRelation;
  relevance: number;
  excerpt: string;
  url: string;
};

export type Verification = {
  id: string;
  image: string;
  caption: string;
  prediction: Prediction;
  confidenceScore: number;
  clipScore?: number;
  alignmentScore?: number;
  reason: string;
  evidence: Evidence[];
  explanation: string;
  inconsistencyType: string;
  createdAt: string;
  dataset: string;
  groundTruthLabel: Prediction | null;
  pipelineStages?: Array<{
    stage: string;
    status: string;
    detail?: string;
    score?: number;
  }>;
  isStub?: boolean;
  configuration?: string;
};

export type ExperimentConfiguration = {
  name: string;
  type: string;
  stages: string[];
  outputs: string[];
  description: string;
  tone: string;
};

export type DatasetSample = {
  id: string;
  dataset: 'NewsCLIPpings' | 'VisualNews';
  image: string;
  caption: string;
  label: Prediction;
  split: 'train' | 'validation' | 'test';
};

export const pipelineStages: { id: PipelineStage; label: string; short: string }[] = [
  { id: 'input_validation', label: 'Input Validation', short: 'Input' },
  { id: 'image_text_analysis', label: 'Image/Text Analysis', short: 'Analyze' },
  { id: 'cross_modal_alignment', label: 'Cross-Modal Alignment', short: 'Align' },
  { id: 'mllm_reasoning', label: 'MLLM Reasoning', short: 'Reason' },
  { id: 'evidence_retrieval', label: 'Evidence Retrieval', short: 'Retrieve' },
  { id: 'evidence_analysis', label: 'Evidence Analysis', short: 'Evaluate' },
  { id: 'final_decision', label: 'Final Decision', short: 'Decide' },
];

export const sampleImages = {
  wildfire: 'https://images.pexels.com/photos/266487/pexels-photo-266487.jpeg?auto=compress&cs=tinysrgb&w=1200',
  assembly: 'https://images.pexels.com/photos/3861969/pexels-photo-3861969.jpeg?auto=compress&cs=tinysrgb&w=1200',
  parliament: 'https://images.pexels.com/photos/466685/pexels-photo-466685.jpeg?auto=compress&cs=tinysrgb&w=1200',
  city: 'https://images.pexels.com/photos/325185/pexels-photo-325185.jpeg?auto=compress&cs=tinysrgb&w=1200',
  river: 'https://images.pexels.com/photos/417074/pexels-photo-417074.jpeg?auto=compress&cs=tinysrgb&w=1200',
  newsroom: 'https://images.pexels.com/photos/518543/pexels-photo-518543.jpeg?auto=compress&cs=tinysrgb&w=1200',
  climate: 'https://images.pexels.com/photos/547115/pexels-photo-547115.jpeg?auto=compress&cs=tinysrgb&w=1200',
  sports: 'https://images.pexels.com/photos/1884574/pexels-photo-1884574.jpeg?auto=compress&cs=tinysrgb&w=1200',
  lab: 'https://images.pexels.com/photos/3786157/pexels-photo-3786157.jpeg?auto=compress&cs=tinysrgb&w=1200',
};

const evidenceForWildfire: Evidence[] = [
  {
    source: 'Associated Press',
    title: 'Fire crews contain the western ridge blaze after overnight winds',
    publishedDate: '14 Jun 2024',
    retrievedDate: '18 Jun 2024',
    relation: 'supports',
    relevance: 0.91,
    excerpt: 'The agency reported that the image was captured near the western ridge during the first evening of the incident.',
    url: 'https://apnews.com/',
  },
  {
    source: 'Regional Fire Service',
    title: 'Incident archive: western ridge response',
    publishedDate: '15 Jun 2024',
    retrievedDate: '18 Jun 2024',
    relation: 'supports',
    relevance: 0.84,
    excerpt: 'Archived updates place the visible smoke plume and response vehicles within the stated area.',
    url: 'https://www.nfpa.org/',
  },
  {
    source: 'Local Desk',
    title: 'Earlier photo recirculated with a new event caption',
    publishedDate: '19 Aug 2023',
    retrievedDate: '18 Jun 2024',
    relation: 'contradicts',
    relevance: 0.63,
    excerpt: 'A visually similar frame was published the previous summer, but the archive does not establish that it is the same image.',
    url: 'https://www.reuters.com/',
  },
];

export const initialMockVerifications: Verification[] = [
  {
    id: 'ver-240618-01',
    image: sampleImages.wildfire,
    caption: 'Fire crews contain a western ridge blaze after overnight winds.',
    prediction: 'Genuine',
    confidenceScore: 0.87,
    reason: 'The caption and visible scene are directionally consistent with the retrieved incident reporting.',
    evidence: evidenceForWildfire,
    explanation: 'The model found agreement between the location language, the visible smoke plume, and two contemporaneous reports. One older visually similar result introduces a small timing ambiguity, so the decision remains reviewable rather than absolute.',
    inconsistencyType: 'No material inconsistency detected',
    createdAt: '18 Jun 2024, 10:42',
    dataset: 'NewsCLIPpings',
    groundTruthLabel: 'Genuine',
    configuration: 'proposed',
    pipelineStages: [
      { stage: 'input_validation', status: 'completed', detail: 'Image format and caption syntax validated.' },
      { stage: 'image_text_analysis', status: 'completed', detail: 'Smoke plume and emergency response entities parsed.' },
      { stage: 'cross_modal_alignment', status: 'completed', detail: 'Semantic similarity score computed.' },
      { stage: 'mllm_reasoning', status: 'completed', detail: 'Multimodal consistency reasoning evaluated.' },
      { stage: 'evidence_retrieval', status: 'completed', detail: 'Retrieved 3 corroborating news reports.' },
      { stage: 'evidence_analysis', status: 'completed', detail: 'Cross-evidence validation confirmed event alignment.' },
      { stage: 'final_decision', status: 'completed', detail: 'Final Genuine prediction synthesized.' },
    ],
  },
  {
    id: 'ver-240617-04',
    image: sampleImages.parliament,
    caption: 'The parliament votes today on the emergency housing package.',
    prediction: 'Misleading',
    confidenceScore: 0.74,
    reason: 'The image is a broad parliamentary session, but the available caption anchors it to a vote not supported by the retrieved context.',
    evidence: [
      {
        source: 'Public Ledger',
        title: 'Emergency housing package returns to committee',
        publishedDate: '17 Jun 2024',
        retrievedDate: '18 Jun 2024',
        relation: 'contradicts',
        relevance: 0.88,
        excerpt: 'The proposal was returned to committee and no floor vote was scheduled for the stated date.',
        url: 'https://www.reuters.com/',
      },
      {
        source: 'Parliament archive',
        title: 'Daily agenda — 17 June 2024',
        publishedDate: '16 Jun 2024',
        retrievedDate: '18 Jun 2024',
        relation: 'insufficient',
        relevance: 0.59,
        excerpt: 'The agenda confirms a sitting but does not identify the pictured chamber or motion.',
        url: 'https://www.parliament.uk/',
      },
    ],
    explanation: 'The visual is plausible for the institution, but the caption makes a specific temporal and procedural claim. Evidence retrieved for that claim does not align with the described vote.',
    inconsistencyType: 'Temporal / event mismatch',
    createdAt: '17 Jun 2024, 16:08',
    dataset: 'VisualNews',
    groundTruthLabel: 'Misleading',
    configuration: 'proposed',
    pipelineStages: [
      { stage: 'input_validation', status: 'completed', detail: 'Image format and caption syntax validated.' },
      { stage: 'image_text_analysis', status: 'completed', detail: 'Parliamentary chamber and voting assertions extracted.' },
      { stage: 'cross_modal_alignment', status: 'completed', detail: 'Cross-modal coherence evaluated.' },
      { stage: 'mllm_reasoning', status: 'completed', detail: 'Temporal and procedural claim divergence identified.' },
      { stage: 'evidence_retrieval', status: 'completed', detail: 'Retrieved 2 archival hearing transcripts.' },
      { stage: 'evidence_analysis', status: 'completed', detail: 'Contradiction identified regarding vote schedule.' },
      { stage: 'final_decision', status: 'completed', detail: 'Final Misleading prediction synthesized.' },
    ],
  },
  {
    id: 'ver-240616-02',
    image: sampleImages.city,
    caption: 'A new transit line opens in the city centre this weekend.',
    prediction: 'Genuine',
    confidenceScore: 0.69,
    reason: 'The urban context is compatible with the caption, while the available sources only partially corroborate the opening date.',
    evidence: [
      {
        source: 'Metro Journal',
        title: 'City transport authority announces trial service',
        publishedDate: '12 Jun 2024',
        retrievedDate: '18 Jun 2024',
        relation: 'supports',
        relevance: 0.72,
        excerpt: 'The authority described a limited trial service beginning over the weekend.',
        url: 'https://www.bbc.com/',
      },
    ],
    explanation: 'The caption is consistent with a transport announcement, though the image itself does not identify the line. This is a low-specificity match and should be read with the evidence panel.',
    inconsistencyType: 'Low visual specificity',
    createdAt: '16 Jun 2024, 09:26',
    dataset: 'NewsCLIPpings',
    groundTruthLabel: 'Genuine',
    configuration: 'proposed',
    pipelineStages: [
      { stage: 'input_validation', status: 'completed', detail: 'Image and caption syntax validated.' },
      { stage: 'image_text_analysis', status: 'completed', detail: 'Urban landscape and transit assertions parsed.' },
      { stage: 'cross_modal_alignment', status: 'completed', detail: 'Semantic alignment measured.' },
      { stage: 'mllm_reasoning', status: 'completed', detail: 'Visual context reasoning completed.' },
      { stage: 'evidence_retrieval', status: 'completed', detail: 'Retrieved 1 corroborating announcement.' },
      { stage: 'evidence_analysis', status: 'completed', detail: 'Supporting evidence relevance evaluated.' },
      { stage: 'final_decision', status: 'completed', detail: 'Final Genuine prediction synthesized.' },
    ],
  },
];

export const datasetSamples: DatasetSample[] = [
  { id: 'nc-001', dataset: 'NewsCLIPpings', image: sampleImages.wildfire, caption: 'Fire crews contain a western ridge blaze after overnight winds.', label: 'Genuine', split: 'test' },
  { id: 'nc-014', dataset: 'NewsCLIPpings', image: sampleImages.assembly, caption: 'Workers inspect a newly unveiled battery assembly line.', label: 'Genuine', split: 'validation' },
  { id: 'nc-028', dataset: 'NewsCLIPpings', image: sampleImages.river, caption: 'The river reaches historic levels after a week of rain.', label: 'Misleading', split: 'train' },
  { id: 'nc-042', dataset: 'NewsCLIPpings', image: sampleImages.climate, caption: 'Arctic researchers record unprecedented ice shelf retreat during summer mission.', label: 'Genuine', split: 'test' },
  { id: 'nc-055', dataset: 'NewsCLIPpings', image: sampleImages.sports, caption: 'Fans celebrate the decisive championship goal during final minutes.', label: 'Genuine', split: 'validation' },
  { id: 'vn-103', dataset: 'VisualNews', image: sampleImages.parliament, caption: 'The parliament votes today on the emergency housing package.', label: 'Misleading', split: 'test' },
  { id: 'vn-118', dataset: 'VisualNews', image: sampleImages.city, caption: 'A new transit line opens in the city centre this weekend.', label: 'Genuine', split: 'validation' },
  { id: 'vn-144', dataset: 'VisualNews', image: sampleImages.newsroom, caption: 'Editors prepare the morning edition before sunrise.', label: 'Genuine', split: 'train' },
  { id: 'vn-172', dataset: 'VisualNews', image: sampleImages.lab, caption: 'Virologists discover synthetic compound effective against airborne pathogens.', label: 'Misleading', split: 'test' },
];

export const experimentConfigurations: ExperimentConfiguration[] = [
  {
    name: 'Alignment-only',
    type: 'Baseline',
    stages: ['Input Validation', 'Image/Text Analysis', 'Cross-Modal Alignment'],
    outputs: ['Alignment signal', 'Pair-level decision'],
    description: 'A compact image-caption comparison that stops before external evidence is introduced.',
    tone: 'violet',
  },
  {
    name: 'MLLM-only',
    type: 'Reasoning baseline',
    stages: ['Input Validation', 'Image/Text Analysis', 'MLLM Reasoning', 'Final Decision'],
    outputs: ['Reasoning trace', 'Pair-level decision'],
    description: 'A multimodal reasoning path that uses the paired input without a retrieval pass.',
    tone: 'blue',
  },
  {
    name: 'Proposed Evidence-aware Pipeline',
    type: 'Full pipeline',
    stages: ['Input Validation', 'Image/Text Analysis', 'Cross-Modal Alignment', 'MLLM Reasoning', 'Evidence Retrieval', 'Evidence Analysis', 'Final Decision'],
    outputs: ['Prediction', 'Decision Confidence', 'Reason', 'Evidence set', 'Explanation'],
    description: 'The complete staged workflow for inspecting how retrieved context changes an explainable decision.',
    tone: 'cyan',
  },
];

// Persistent storage and reactive updates
const STORAGE_KEY = 'ooc_verify_user_verifications';
const subscribers = new Set<() => void>();

function getStoredVerifications(): Verification[] {
  if (typeof window === 'undefined') return [...initialMockVerifications];
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed) && parsed.length > 0) {
        return parsed;
      }
    }
  } catch (err) {
    console.warn('Could not read stored verifications:', err);
  }
  return [...initialMockVerifications];
}

let activeVerifications: Verification[] = getStoredVerifications();

function saveVerifications(items: Verification[]) {
  activeVerifications = items;
  if (typeof window !== 'undefined') {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(items));
    } catch (err) {
      console.warn('Could not persist verifications:', err);
    }
  }
  subscribers.forEach((callback) => callback());
}

export const mockService = {
  subscribe(callback: () => void): () => void {
    subscribers.add(callback);
    return () => subscribers.delete(callback);
  },

  listVerifications(): Verification[] {
    return [...activeVerifications];
  },

  getVerification(id: string): Verification | undefined {
    return activeVerifications.find((verification) => verification.id === id);
  },

  addVerification(verification: Verification): void {
    const exists = activeVerifications.some((v) => v.id === verification.id);
    const updated = exists
      ? activeVerifications.map((v) => (v.id === verification.id ? verification : v))
      : [verification, ...activeVerifications];
    saveVerifications(updated);
  },

  listDatasetSamples(): DatasetSample[] {
    return [...datasetSamples];
  },

  getDatasetSample(id: string): DatasetSample | undefined {
    return datasetSamples.find((sample) => sample.id === id);
  },

  getExperimentConfigurations(): ExperimentConfiguration[] {
    return experimentConfigurations;
  },

  deleteVerification(id: string): void {
    const filtered = activeVerifications.filter((item) => item.id !== id);
    saveVerifications(filtered.length > 0 ? filtered : [...initialMockVerifications]);
  },

  resetVerifications(): void {
    saveVerifications([...initialMockVerifications]);
  },

  createVerification(input: {
    image: string;
    caption: string;
    dataset: string;
    backendResult: {
      id?: string;
      prediction: Prediction;
      confidence_score?: number;
      confidenceScore?: number;
      clip_score?: number;
      clipScore?: number;
      alignment_score?: number;
      alignmentScore?: number;
      reason: string;
      evidence?: Evidence[];
      explanation: string;
      inconsistency_type?: string;
      inconsistencyType?: string;
      dataset?: string;
      ground_truth_label?: Prediction | null;
      groundTruthLabel?: Prediction | null;
      pipeline_stages?: Array<{
        stage: string;
        status: string;
        detail?: string;
        score?: number;
      }>;
      is_stub?: boolean;
      isStub?: boolean;
      configuration?: string;
    };
    configuration?: string;
  }): Verification {
    const now = new Date();

    const formattedDate = now.toLocaleDateString('en-GB', {
      day: '2-digit',
      month: 'short',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });

    const backend = input.backendResult;

    const verification: Verification = {
      id:
        backend.id ||
        `ver-${Date.now().toString(36).toUpperCase()}-${Math.floor(
          100 + Math.random() * 900
        )}`,

      image: input.image,

      caption: input.caption.trim(),

      prediction: backend.prediction,

      confidenceScore:
        backend.confidence_score ??
        backend.confidenceScore ??
        0,

      clipScore:
        backend.clip_score ??
        backend.clipScore,

      alignmentScore:
        backend.alignment_score ??
        backend.alignmentScore ??
        backend.clip_score ??
        backend.clipScore,

      reason: backend.reason,

      evidence: backend.evidence ?? [],

      explanation: backend.explanation,

      inconsistencyType:
        backend.inconsistency_type ??
        backend.inconsistencyType ??
        'Evidence-aware multimodal decision',

      createdAt: formattedDate,

      dataset:
        backend.dataset ||
        input.dataset ||
        'Custom Pair',

      groundTruthLabel:
        backend.ground_truth_label ??
        backend.groundTruthLabel ??
        null,

      pipelineStages:
        backend.pipeline_stages?.map((stage) => ({
          stage: stage.stage,
          status: stage.status,
          detail: stage.detail,
          score: stage.score,
        })),

      isStub:
        backend.is_stub ??
        backend.isStub ??
        false,

      configuration:
        backend.configuration ||
        input.configuration ||
        'proposed',
    };

    saveVerifications([
      verification,
      ...activeVerifications,
    ]);

    return verification;
  },
};

// React hook for consuming verifications with reactive synchronization
export function useVerifications() {
  const [items, setItems] = useState<Verification[]>(() => mockService.listVerifications());

  useEffect(() => {
    return mockService.subscribe(() => {
      setItems(mockService.listVerifications());
    });
  }, []);

  return items;
}