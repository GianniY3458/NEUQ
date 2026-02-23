import { useState, useEffect, useRef, useCallback } from 'react';
import { searchGallery, getStats, uploadGallery, clearGallery } from '../api/ssan';
import type { SearchItem, StatsResponse } from '../types/ssan';

/**
 * SSAN 页面核心业务逻辑 hook
 * 将搜索 / 上传 / 清空 / 状态查询全部封装，视图层只管渲染。
 */
export default function useSSAN() {
  /* ---- 搜索 ---- */
  const [text, setText] = useState('');
  const [topk, setTopk] = useState(5);
  const [results, setResults] = useState<SearchItem[]>([]);
  const [searchMs, setSearchMs] = useState<number | null>(null);

  /* ---- 图库状态 ---- */
  const [stats, setStats] = useState<StatsResponse | null>(null);

  /* ---- 上传 ---- */
  const [uploadFiles, setUploadFiles] = useState<File[]>([]);
  const [uploadMsg, setUploadMsg] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  /* ---- 通用 ---- */
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // 页面加载 → 获取图库状态
  useEffect(() => {
    fetchStats();
  }, []);

  // 释放 ObjectURL，防止内存泄漏
  useEffect(() => {
    const urls = uploadFiles.map(f => URL.createObjectURL(f));
    return () => urls.forEach(u => URL.revokeObjectURL(u));
  }, [uploadFiles]);

  /* ---------- 图库状态 ---------- */
  const fetchStats = useCallback(async () => {
    try {
      const data = await getStats();
      setStats(data);
      setError('');
    } catch (e: any) {
      setError('获取图库状态失败: ' + (e.response?.data?.detail || e.message));
    }
  }, []);

  /* ---------- 搜索 ---------- */
  const handleSearch = useCallback(async () => {
    if (!text.trim()) return;
    setLoading(true);
    setError('');
    try {
      const resp = await searchGallery({ text, topk });
      setResults(resp.results);
      setSearchMs(resp.ms);
    } catch (e: any) {
      setResults([]);
      setSearchMs(null);
      setError('搜索失败: ' + (e.response?.data?.detail || e.message));
    } finally {
      setLoading(false);
    }
  }, [text, topk]);

  /* ---------- 文件选择 ---------- */
  const handleFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setUploadFiles(Array.from(e.target.files));
      setUploadMsg('');
    }
  }, []);

  /* ---------- 上传 ---------- */
  const handleUpload = useCallback(async () => {
    if (uploadFiles.length === 0) {
      setUploadMsg('请先选择图片文件');
      return;
    }
    setLoading(true);
    setError('');
    setUploadMsg('');
    try {
      const resp = await uploadGallery(uploadFiles);
      setUploadMsg(resp.message);
      setUploadFiles([]);
      if (fileInputRef.current) fileInputRef.current.value = '';
      await fetchStats();
    } catch (e: any) {
      setError('上传失败: ' + (e.response?.data?.detail || e.message));
    } finally {
      setLoading(false);
    }
  }, [uploadFiles, fetchStats]);

  /* ---------- 清空 ---------- */
  const handleClear = useCallback(async () => {
    if (!window.confirm('确定要清空图库吗？此操作不可恢复。')) return;
    setLoading(true);
    setError('');
    try {
      await clearGallery();
      setResults([]);
      setSearchMs(null);
      setUploadMsg('');
      await fetchStats();
    } catch (e: any) {
      setError('清空失败: ' + (e.response?.data?.detail || e.message));
    } finally {
      setLoading(false);
    }
  }, [fetchStats]);

  return {
    // 状态
    text, topk, results, searchMs,
    stats, uploadFiles, uploadMsg,
    loading, error, fileInputRef,
    // 操作
    setText, setTopk,
    fetchStats, handleSearch,
    handleFileChange, handleUpload, handleClear,
  };
}
