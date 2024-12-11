import React, { useState, useRef, useEffect } from 'react';
import styled from 'styled-components';

const CropperContainer = styled.div`
  position: relative;
  max-width: 500px;
  margin: 0 auto;
  user-select: none;
`;

const ImageContainer = styled.div`
  position: relative;
  overflow: hidden;
  width: 100%;
  aspect-ratio: 1;
  background: #f0f0f0;
  cursor: move;
`;

const CropImage = styled.img`
  position: absolute;
  max-width: none;
  transform-origin: top left;
  transition: none;
`;

const CropOverlay = styled.div`
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  border: 2px solid #fff;
  box-shadow: 0 0 0 9999px rgba(0, 0, 0, 0.5);
`;

const CropControls = styled.div`
  margin-top: 1rem;
  display: flex;
  gap: 1rem;
  justify-content: center;
`;

const Button = styled.button`
  background: #4299e1;
  color: white;
  border: none;
  padding: 0.5rem 1rem;
  border-radius: 4px;
  cursor: pointer;
  
  &:hover {
    background: #3182ce;
  }
`;

const ImageCropper = ({ file, onCropComplete, onCancel }) => {
  const [scale, setScale] = useState(1);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [dragStart, setDragStart] = useState(null);
  const [image, setImage] = useState(null);
  const imageRef = useRef(null);
  const containerRef = useRef(null);

  useEffect(() => {
    if (file) {
      const url = URL.createObjectURL(file);
      const img = new Image();
      img.onload = () => {
        setImage({
          url,
          width: img.width,
          height: img.height
        });
        
        // Calculate initial scale to fit image
        const container = containerRef.current;
        if (container) {
          const containerSize = container.offsetWidth;
          const scale = containerSize / Math.max(img.width, img.height);
          setScale(scale);
          
          // Center image
          const scaledWidth = img.width * scale;
          const scaledHeight = img.height * scale;
          setPosition({
            x: (containerSize - scaledWidth) / 2,
            y: (containerSize - scaledHeight) / 2
          });
        }
      };
      img.src = url;
      return () => URL.revokeObjectURL(url);
    }
  }, [file]);

  const handleMouseDown = (e) => {
    e.preventDefault();
    setDragStart({
      x: e.clientX - position.x,
      y: e.clientY - position.y
    });
  };

  const handleMouseMove = (e) => {
    if (dragStart) {
      setPosition({
        x: e.clientX - dragStart.x,
        y: e.clientY - dragStart.y
      });
    }
  };

  const handleMouseUp = () => {
    setDragStart(null);
  };

  const handleWheel = (e) => {
    e.preventDefault();
    const delta = e.deltaY * -0.01;
    const newScale = Math.max(0.1, Math.min(10, scale + delta));
    setScale(newScale);
  };

  const cropImage = () => {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const container = containerRef.current;
    const containerSize = container.offsetWidth;
    
    // Set canvas size to desired output size (224x224 or your preferred size)
    canvas.width = 224;
    canvas.height = 224;
    
    // Calculate crop area in original image coordinates
    const scaleX = image.width / (containerSize / scale);
    const scaleY = image.height / (containerSize / scale);
    
    ctx.drawImage(
      imageRef.current,
      -position.x * scaleX / scale,
      -position.y * scaleY / scale,
      containerSize * scaleX,
      containerSize * scaleY,
      0,
      0,
      224,
      224
    );
    
    canvas.toBlob((blob) => {
      onCropComplete(new File([blob], 'cropped.jpg', { type: 'image/jpeg' }));
    }, 'image/jpeg');
  };

  if (!image) return null;

  return (
    <CropperContainer>
      <ImageContainer
        ref={containerRef}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onWheel={handleWheel}
      >
        <CropImage
          ref={imageRef}
          src={image.url}
          style={{
            transform: `translate(${position.x}px, ${position.y}px) scale(${scale})`,
            width: image.width,
            height: image.height
          }}
          draggable="false"
        />
        <CropOverlay />
      </ImageContainer>
      <CropControls>
        <Button onClick={cropImage}>Crop Image</Button>
        <Button onClick={onCancel}>Cancel</Button>
      </CropControls>
    </CropperContainer>
  );
};

export default ImageCropper;