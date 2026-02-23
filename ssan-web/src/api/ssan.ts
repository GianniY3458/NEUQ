import apiClient from './client';
import type { SearchRequest, SearchResponse, StatsResponse, UploadResponse } from '../types/ssan';

// 搜索接口
export async function searchGallery(params: SearchRequest): Promise<SearchResponse> {
  const resp = await apiClient.post<SearchResponse>('/search', params);
  return resp.data;
}

// 获取图库统计
export async function getStats(): Promise<StatsResponse> {
  const resp = await apiClient.get<StatsResponse>('/stats');
  return resp.data;
}

// 上传图片并更新图库
export async function uploadGallery(files: File[]): Promise<UploadResponse> {
  const formData = new FormData();
  files.forEach(file => formData.append('files', file));
  const resp = await apiClient.post<UploadResponse>('/upload_gallery', formData);
  return resp.data;
}

// 清空图库
export async function clearGallery(): Promise<{ success: boolean; message: string }> {
  const resp = await apiClient.post<{ success: boolean; message: string }>('/clear_gallery');
  return resp.data;
}