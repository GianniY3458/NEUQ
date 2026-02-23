import { useRef, useState, useCallback } from 'react';
import './ImageUploader.css';

interface ImageUploaderProps {
  label: string;
  onFileSelect: (file: File | undefined) => void;
  disabled?: boolean;
}

export default function ImageUploader({ label, onFileSelect, disabled }: ImageUploaderProps) {
  const [preview, setPreview] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(
    (file: File | undefined) => {
      if (file && file.type.startsWith('image/')) {
        const url = URL.createObjectURL(file);
        setPreview(url);
        onFileSelect(file);
      }
    },
    [onFileSelect]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const file = e.dataTransfer.files[0];
      handleFile(file);
    },
    [handleFile]
  );

  const handleClear = useCallback(() => {
    setPreview(null);
    onFileSelect(undefined);
    if (inputRef.current) inputRef.current.value = '';
  }, [onFileSelect]);

  return (
    <div
      className={`image-uploader ${dragOver ? 'image-uploader--dragover' : ''} ${disabled ? 'image-uploader--disabled' : ''}`}
      onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
      onDragLeave={() => setDragOver(false)}
      onDrop={handleDrop}
      onClick={() => !disabled && inputRef.current?.click()}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="image-uploader-input"
        onChange={(e) => handleFile(e.target.files?.[0])}
        disabled={disabled}
      />
      {preview ? (
        <div className="image-uploader-preview">
          <img src={preview} alt={label} />
          <button
            className="image-uploader-clear"
            onClick={(e) => { e.stopPropagation(); handleClear(); }}
          >
            X
          </button>
        </div>
      ) : (
        <div className="image-uploader-placeholder">
          <span className="image-uploader-label">{label}</span>
          <span className="image-uploader-hint">Drop or click</span>
        </div>
      )}
    </div>
  );
}
