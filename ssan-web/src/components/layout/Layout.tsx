import { Outlet } from 'react-router-dom';
import TopNav from './TopNav';
import './Layout.css';

export default function Layout() {
  return (
    <div className="layout">
      <TopNav />
      <main className="layout-main">
        <Outlet />
      </main>
    </div>
  );
}
