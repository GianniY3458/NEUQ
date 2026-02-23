import axios from 'axios';

/** 后端 API 基础地址，图片 URL 拼接也用此常量 */
export const API_BASE = 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
});

export default apiClient;