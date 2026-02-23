import { NavLink } from 'react-router-dom';
import './TopNav.css';

const tabs = [
  { label: 'SSAN', sublabel: 'Text-to-Image', path: '/ssan' },
  { label: 'CSIP', sublabel: 'Video Retrieval', path: '/csip' },
  { label: 'ReID5o', sublabel: 'Multi-Modal', path: '/reid5o' },
  { label: 'Module 4', sublabel: 'TBD', path: '/tab4' },
  { label: 'Module 5', sublabel: 'TBD', path: '/tab5' },
];

export default function TopNav() {
  return (
    <nav className="topnav">
      <div className="topnav-brand">
        <span className="topnav-title">Multi-Modal ReID</span>
      </div>
      <div className="topnav-tabs">
        {tabs.map((tab) => (
          <NavLink
            key={tab.path}
            to={tab.path}
            className={({ isActive }) =>
              `topnav-tab ${isActive ? 'topnav-tab--active' : ''}`
            }
          >
            <span className="topnav-tab-label">{tab.label}</span>
            <span className="topnav-tab-sublabel">{tab.sublabel}</span>
          </NavLink>
        ))}
      </div>
    </nav>
  );
}
