import {
  type Verification,
  type Evidence,
  type Prediction,
  mockService,
} from './mock-data';

// Default to environment variable, then fallback to deployed Render backend
const API_BASE_URL = 'https://ooc-verify.onrender.com';

export interface BackendPipelineStage {
  stage: string;
  status: string;
  detail?: string;
  score?: number;
}

export interface BackendVerificationResponse {
  id: string;
  prediction: string;
  confidence_score: number;
  confidenceScore?: number;
  clip_score?: number;
  clipScore?: number;
  alignment_score?: number;
  alignmentScore?: number;
  reason: string;
  evidence: Evidence[];
  explanation: string;
  inconsistency_type: string;
  inconsistencyType?: string;
  dataset: string;
  ground_truth_label: string | null;
  groundTruthLabel?: string | null;
  pipeline_stages: BackendPipelineStage[];
  is_stub?: boolean;
  configuration?: string;
}

export interface VerifyRequestParams {
  imageFile?: File | null;
  imageUrl?: string;
  caption: string;
  dataset?: string;
  configuration?: 'alignment_only' | 'mllm_only' | 'proposed' | string;
}

/**
 * Helper to convert an image URL or base64 Data URL to a File object for multipart upload.
 */
async function resolveImageFile(
  file?: File | null,
  imageUrl?: string,
): Promise<File> {
  if (file instanceof File) {
    return file;
  }

  const fallbackName = 'verification_input.jpg';

  if (imageUrl && imageUrl.startsWith('data:')) {
    const res = await fetch(imageUrl);
    const blob = await res.blob();
    const type = blob.type || 'image/jpeg';
    return new File([blob], fallbackName, { type });
  }

  if (imageUrl && imageUrl.startsWith('http')) {
    try {
      const res = await fetch(imageUrl, { mode: 'cors' });
      if (res.ok) {
        const blob = await res.blob();
        return new File([blob], fallbackName, {
          type: blob.type || 'image/jpeg',
        });
      }
    } catch (err) {
      console.warn(
        'Could not fetch remote image directly, creating canvas/data representation:',
        err,
      );
    }
  }

  // Fallback valid minimal JPEG if remote image blocked by CORS
  const minimalJpeg = new Uint8Array([
    0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10, 0x4a, 0x46, 0x49, 0x46, 0x00, 0x01,
    0x01, 0x01, 0x00, 0x48, 0x00, 0x48, 0x00, 0x00, 0xff, 0xdb, 0x00, 0x43,
    0x00, 0x08, 0x06, 0x06, 0x07, 0x06, 0x05, 0x08, 0x07, 0x07, 0x07, 0x09,
    0x09, 0x08, 0x0a, 0x0c, 0x14, 0x0d, 0x0c, 0x0b, 0x0b, 0x0c, 0x19, 0x12,
    0x13, 0x0f, 0x14, 0x1d, 0x1a, 0x1f, 0x1e, 0x1d, 0x1a, 0x1c, 0x1c, 0x20,
    0x24, 0x2e, 0x27, 0x20, 0x22, 0x2c, 0x23, 0x1c, 0x1c, 0x28, 0x37, 0x29,
    0x2c, 0x30, 0x31, 0x34, 0x34, 0x34, 0x1f, 0x27, 0x39, 0x3d, 0x38, 0x32,
    0x3c, 0x2e, 0x33, 0x34, 0x32, 0xff, 0xc0, 0x00, 0x0b, 0x08, 0x00, 0x01,
    0x00, 0x01, 0x01, 0x01, 0x11, 0x00, 0xff, 0xc4, 0x00, 0x14, 0x00, 0x01,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0xff, 0xda, 0x00, 0x08, 0x01, 0x01, 0x00, 0x00,
    0x3f, 0x00, 0x37, 0xff, 0xd9,
  ]);
  return new File([minimalJpeg], fallbackName, { type: 'image/jpeg' });
}

export const apiService = {
  /**
   * Check backend health status
   */
  async checkHealth(): Promise<{ status: string; service: string }> {
    const res = await fetch(`${API_BASE_URL}/api/v1/health`);
    if (!res.ok) {
      throw new Error(`Health check failed with status: ${res.status}`);
    }
    return res.json();
  },

  /**
   * Execute verification by sending image and caption to FastAPI backend
   */
  async verify(params: VerifyRequestParams): Promise<Verification> {
    const file = await resolveImageFile(params.imageFile, params.imageUrl);

    console.log('OOC-Verify image sent to backend:', {
      name: file.name,
      type: file.type,
      size: file.size,
    });
    const formData = new FormData();
    formData.append('image', file, file.name);
    formData.append('caption', params.caption.trim());
    if (params.dataset) {
      formData.append('dataset', params.dataset);
    }
    if (params.configuration) {
      formData.append('configuration', params.configuration);
    }

    const res = await fetch(`${API_BASE_URL}/api/v1/verify`, {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) {
      let errorMsg = `Server error ${res.status}`;
      try {
        const errorJson = await res.json();
        if (errorJson.detail) {
          errorMsg = errorJson.detail;
        }
      } catch {
        // use default errorMsg
      }
      throw new Error(errorMsg);
    }

    const data: BackendVerificationResponse = await res.json();

    const now = new Date();
    const formattedDate = now.toLocaleDateString('en-GB', {
      day: '2-digit',
      month: 'short',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });

    const verification: Verification = {
      id: data.id,
      image: params.imageUrl || (params.imageFile ? URL.createObjectURL(params.imageFile) : ''),
      caption: params.caption.trim(),
      prediction: (data.prediction || 'Genuine') as Prediction,
      confidenceScore: data.confidence_score ?? data.confidenceScore ?? 0.0,
      clipScore: data.clip_score ?? data.clipScore,
      alignmentScore: data.alignment_score ?? data.alignmentScore ?? data.clip_score ?? data.clipScore,
      reason: data.reason,
      evidence: data.evidence || [],
      explanation: data.explanation,
      inconsistencyType:
        data.inconsistency_type ||
        data.inconsistencyType ||
        'No material inconsistency detected',
      createdAt: formattedDate,
      dataset: data.dataset || params.dataset || 'Custom Pair',
      groundTruthLabel: (data.ground_truth_label || data.groundTruthLabel || null) as Prediction | null,
      pipelineStages: data.pipeline_stages,
      isStub: data.is_stub ?? false,
      configuration: (data as any).configuration || params.configuration || 'proposed',
    };

    // Store in the reactive verification list so other pages (History, Results, Dashboard) reflect it
    mockService.addVerification(verification);

    return verification;
  },

  /**
   * Get a verification report by ID
   */
  getVerification(id: string): Verification | undefined {
    return mockService.getVerification(id);
  },

  /**
   * List all stored verifications
   */
  listVerifications(): Verification[] {
    return mockService.listVerifications();
  },
};
