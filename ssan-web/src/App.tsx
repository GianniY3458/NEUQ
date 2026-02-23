import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import Layout from './components/layout/Layout';
import SSANPage from './pages/SSANPage';
import ComingSoon from './components/common/ComingSoon';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          {/* 默认跳转到 SSAN */}
          <Route index element={<Navigate to="/ssan" replace />} />
          <Route path="/ssan" element={<SSANPage />} />
          <Route path="/csip" element={<ComingSoon title="CSIP — Video Retrieval" />} />
          <Route path="/reid5o" element={<ComingSoon title="ReID5o — Multi-Modal" />} />
          <Route path="/tab4" element={<ComingSoon title="Module 4" />} />
          <Route path="/tab5" element={<ComingSoon title="Module 5" />} />
          {/* 兜底 */}
          <Route path="*" element={<Navigate to="/ssan" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}