import './ComingSoon.css';

interface ComingSoonProps {
  title: string;
}

export default function ComingSoon({ title }: ComingSoonProps) {
  return (
    <div className="coming-soon">
      <div className="coming-soon-inner">
        <h2 className="coming-soon-title">{title}</h2>
        <p className="coming-soon-text">This module is under development.</p>
        <div className="coming-soon-line" />
      </div>
    </div>
  );
}
