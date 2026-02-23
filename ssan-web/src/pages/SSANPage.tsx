import { useMemo } from 'react';
import { API_BASE } from '../api/client';
import useSSAN from '../hooks/useSSAN';
import './SSANPage.css';

export default function SSANPage() {
  const {
    text, topk, results, searchMs,
    stats, uploadFiles, uploadMsg,
    loading, error, fileInputRef,
    setText, setTopk,
    fetchStats, handleSearch,
    handleFileChange, handleUpload, handleClear,
  } = useSSAN();

  // 缩略图 URL（useMemo 防止每次渲染都重新创建）
  const thumbUrls = useMemo(
    () => uploadFiles.map(f => URL.createObjectURL(f)),
    [uploadFiles],
  );

  return (
    <div className="ssan-page">
      {loading && <div className="ssan-loading-overlay">处理中…</div>}

      {/* 图库状态栏 */}
      <div className="ssan-status-bar">
        <span>
          {stats
            ? `图库 ${stats.gallery_size} 张 · 维度 ${stats.feature_dim} · ${stats.device}`
            : '加载中…'}
        </span>
        <button onClick={fetchStats}>刷新</button>
      </div>

      {error && <div className="ssan-error">{error}</div>}

      {/* 搜索 */}
      <section className="ssan-search-section">
        <h2>文本搜索</h2>
        <div className="ssan-search-bar">
          <input
            className="ssan-search-input"
            value={text}
            onChange={e => setText(e.target.value)}
            placeholder="输入描述，如：a man wearing red shirt and black pants"
            onKeyDown={e => e.key === 'Enter' && handleSearch()}
          />
          <div className="ssan-topk-group">
            <span>Top</span>
            <input
              className="ssan-topk-input"
              type="number"
              value={topk}
              onChange={e => setTopk(Math.max(1, Math.min(100, Number(e.target.value))))}
              min={1}
              max={100}
            />
          </div>
          <button
            className="ssan-btn"
            onClick={handleSearch}
            disabled={loading || !text.trim()}
          >
            搜索
          </button>
        </div>
      </section>

      {/* 搜索结果 */}
      <section className="ssan-results-section">
        {searchMs !== null && (
          <div className="ssan-results-meta">
            {results.length} 条结果 · {searchMs.toFixed(1)} ms
          </div>
        )}

        {results.length > 0 ? (
          <div className="ssan-results-grid">
            {results.map((item, idx) => (
              <div className="ssan-result-card" key={idx}>
                <img
                  className="ssan-result-img"
                  src={`${API_BASE}${item.image_url}`}
                  alt={`rank-${idx + 1}`}
                  loading="lazy"
                  onError={e => {
                    (e.target as HTMLImageElement).style.opacity = '0.3';
                  }}
                />
                <div className="ssan-result-info">
                  <span className="ssan-result-rank">#{idx + 1}</span>
                  <span className="ssan-result-score">{item.score.toFixed(4)}</span>
                </div>
              </div>
            ))}
          </div>
        ) : searchMs !== null ? (
          <div className="ssan-no-results">没有匹配的结果</div>
        ) : null}
      </section>

      {/* 上传 */}
      <section className="ssan-upload-section">
        <h2>上传图片到图库</h2>
        <div className="ssan-upload-row">
          <input
            ref={fileInputRef}
            className="ssan-file-input"
            type="file"
            multiple
            accept="image/*"
            onChange={handleFileChange}
          />
          <button
            className="ssan-btn"
            onClick={handleUpload}
            disabled={loading || uploadFiles.length === 0}
          >
            上传 ({uploadFiles.length})
          </button>
        </div>

        {thumbUrls.length > 0 && (
          <div className="ssan-upload-previews">
            {thumbUrls.map((url, i) => (
              <img key={i} className="ssan-upload-thumb" src={url} alt="" />
            ))}
          </div>
        )}

        {uploadMsg && <div className="ssan-upload-msg">{uploadMsg}</div>}
      </section>

      {/* 底部操作 */}
      <div className="ssan-actions">
        <button
          className="ssan-btn-danger"
          onClick={handleClear}
          disabled={loading}
        >
          清空图库
        </button>
      </div>
    </div>
  );
}
