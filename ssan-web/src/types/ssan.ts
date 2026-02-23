export interface SearchRequest {
  text: string;
  topk: number;
}

export interface SearchItem {
  path: string;
  image_url: string;   // 后端返回的可直接使用的图片 URL
  score: number;
}

export interface SearchResponse {
  text: string;
  topk: number;
  ms: number;
  results: SearchItem[];
}

export interface StatsResponse {
  gallery_size: number;
  feature_dim: number;
  device: string;
}

export interface UploadResponse {
  success: boolean;
  message: string;
  uploaded_files: string[];
}