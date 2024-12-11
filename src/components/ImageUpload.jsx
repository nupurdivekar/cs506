import React, { useState, useRef, useEffect } from 'react';
import styled from 'styled-components';

const UploadContainer = styled.div`
    border: 2px dashed #cbd5e0;
    border-radius: 8px;
    padding: 2rem;
    text-align: center;
    background-color: #f7fafc;
    cursor: pointer;
    margin-bottom: 1rem;

    &:hover {
        background-color: #edf2f7;
    }
`;

const ImageEditorContainer = styled.div`
    position: relative;
    max-width: 800px;
    margin: 0 auto;
`;

const ImageWrapper = styled.div`
    position: relative;
    width: 100%;
    background: #f0f0f0;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2), 0 4px 6px -2px rgba(0, 0, 0, 0.15);
    border-radius: 8px;
    overflow: hidden;
`;

const MainImage = styled.img`
    width: 100%;
    display: block;
    user-select: none;
    -webkit-user-drag: none;
`;

const CropSquare = styled.div`
    position: absolute;
    border: 2px solid white;
    box-shadow: 0 0 0 9999px rgba(0, 0, 0, 0.5);
    cursor: move;
    background: transparent;
`;

const ResizeHandle = styled.div`
    position: absolute;
    right: -6px;
    bottom: -6px;
    width: 12px;
    height: 12px;
    background: white;
    border: 1px solid #666;
    border-radius: 2px;
    cursor: se-resize;
    z-index: 10;
`;

const PreviewContainer = styled.div`
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    margin: 1rem 0;
`;

const PreviewImage = styled.img`
    width: 244px;
    height: 244px;
    object-fit: cover;
    border: 1px solid #ccc;
    margin-top: 0.5rem;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2), 0 4px 6px -2px rgba(0, 0, 0, 0.15);
`;

const ButtonContainer = styled.div`
    display: flex;
    gap: 1rem;
    margin: 1rem 0;
    justify-content: flex-end;
`;

const Caption = styled.h3`
    text-align: center;
    margin-bottom: 0.5rem;
    color: #4a5568;
`;

const Button = styled.button`
    padding: 0.5rem 1rem;
    background: #4299e1;
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    
    &:disabled {
        background: #a0aec0;
        cursor: not-allowed;
    }
    
    &:hover:not(:disabled) {
        background: #3182ce;
    }
`;

const ErrorMessage = styled.div`
    color: #e53e3e;
    margin: 1rem 0;
`;

const ImageUpload = ({ onUpload }) => {
    const [image, setImage] = useState(null);
    const [error, setError] = useState(null);
    const [cropArea, setCropArea] = useState({ x: 0, y: 0, size: 300 });
    const [isDragging, setIsDragging] = useState(false);
    const [isResizing, setIsResizing] = useState(false);
    const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
    const [croppedPreview, setCroppedPreview] = useState(null);
    const [croppedImage, setCroppedImage] = useState(null);
    
    const imageRef = useRef(null);
    const fileInputRef = useRef(null);

    const handleFileSelect = (event) => {
        const file = event.target.files[0];
        if (file) {
        setError(null);
        if (croppedPreview) {
            URL.revokeObjectURL(croppedPreview);
            setCroppedPreview(null);
        }
        setCroppedImage(null);

        const tempURL = URL.createObjectURL(file);
        const img = new Image();
        
        img.onload = () => {
            URL.revokeObjectURL(tempURL);

            if (img.width < 244 || img.height < 244) {
            setError('Image must be at least 244x244 pixels');
            setImage(null);
            return;
            }

            setImage({
            file,
            url: URL.createObjectURL(file),
            width: img.width,
            height: img.height
            });

            const initialSize = 300;
            setCropArea({
            x: Math.max(0, (img.width - initialSize) / 2),
            y: Math.max(0, (img.height - initialSize) / 2),
            size: Math.min(initialSize, img.width, img.height)
            });
        };

        img.onerror = () => {
            URL.revokeObjectURL(tempURL);
            setError('Failed to load image. Please try another file.');
            setImage(null);
        };

        img.src = tempURL;
        }
    };

    const handleMouseDown = (e, action) => {
        if (croppedPreview) return;
        e.stopPropagation();
        const bounds = imageRef.current.getBoundingClientRect();
        const scale = image.width / bounds.width;
        
        if (action === 'resize') {
        setIsResizing(true);
        setIsDragging(false);
        } else {
        setIsDragging(true);
        setIsResizing(false);
        }
        
        setDragStart({
        x: e.clientX,
        y: e.clientY,
        initialX: cropArea.x,
        initialY: cropArea.y,
        initialSize: cropArea.size
        });
    };

    const handleMouseMove = (e) => {
        if (croppedPreview || (!isDragging && !isResizing)) return;
        
        const bounds = imageRef.current.getBoundingClientRect();
        const scale = image.width / bounds.width;
        
        if (isResizing) {
        const dx = (e.clientX - dragStart.x) * scale;
        const dy = (e.clientY - dragStart.y) * scale;
        const delta = Math.max(dx, dy);
        
        const newSize = Math.max(
            244,
            Math.min(
            dragStart.initialSize + delta,
            image.width - dragStart.initialX,
            image.height - dragStart.initialY
            )
        );
        
        setCropArea({
            x: dragStart.initialX,
            y: dragStart.initialY,
            size: newSize
        });
        } else if (isDragging) {
        const dx = (e.clientX - dragStart.x) * scale;
        const dy = (e.clientY - dragStart.y) * scale;
        
        const newX = Math.max(0, Math.min(dragStart.initialX + dx, image.width - cropArea.size));
        const newY = Math.max(0, Math.min(dragStart.initialY + dy, image.height - cropArea.size));
        
        setCropArea(prev => ({
            ...prev,
            x: newX,
            y: newY
        }));
        }
    };

    const handleMouseUp = () => {
        setIsDragging(false);
        setIsResizing(false);
    };

    const handleCrop = () => {
        if (croppedPreview) return;
            
        console.log('Starting crop with:', {
            x: cropArea.x,
            y: cropArea.y,
            size: cropArea.size
        });

        const canvas = document.createElement('canvas');
        canvas.width = 244;
        canvas.height = 244;
        const ctx = canvas.getContext('2d');
            
        ctx.drawImage(
            imageRef.current,
            cropArea.x,
            cropArea.y,
            cropArea.size,
            cropArea.size,
            0,
            0,
            244,
            244
        );
            
        canvas.toBlob((blob) => {
            if (!blob) {
                console.error('Failed to create blob');
                return;
            }
            console.log('Blob created successfully');
            const croppedFile = new File([blob], 'cropped.jpg', { type: 'image/jpeg' });
            const previewUrl = URL.createObjectURL(blob);
            setCroppedPreview(previewUrl);
            setCroppedImage(croppedFile);
            console.log('Crop completed successfully');
        }, 'image/jpeg', 0.95);
    };

    const handleReCrop = () => {
        console.log('Starting re-crop');
        if (croppedPreview) {
        console.log('Revoking old preview URL');
        URL.revokeObjectURL(croppedPreview);
        }
        setCroppedPreview(null);
        setCroppedImage(null);
        console.log('Re-crop complete');
    };

    const handleTest = () => {
        if (croppedImage) {
        onUpload(croppedImage);
        }
    };

    useEffect(() => {
        return () => {
        if (image?.url) URL.revokeObjectURL(image.url);
        if (croppedPreview) URL.revokeObjectURL(croppedPreview);
        };
    }, [image, croppedPreview]);

    return (
        <div>
            {!image && (
                <UploadContainer onClick={() => fileInputRef.current.click()}>
                    <input
                        type="file"
                        ref={fileInputRef}
                        onChange={handleFileSelect}
                        accept="image/*"
                        style={{ display: 'none' }}
                    />
                    <p>Click this box to upload an image</p>
                    <p>Minimum size: 244x244 pixels</p>
                </UploadContainer>
            )}
    
            {error && <ErrorMessage>{error}</ErrorMessage>}
    
            {image && (
                <ImageEditorContainer
                    onMouseMove={!croppedPreview ? handleMouseMove : undefined}
                    onMouseUp={!croppedPreview ? handleMouseUp : undefined}
                    onMouseLeave={!croppedPreview ? handleMouseUp : undefined}
                >
                    <Caption>Uploaded Image</Caption>
                    <ImageWrapper>
                        <MainImage ref={imageRef} src={image.url} alt="Upload" />
                        {!croppedPreview && (
                            <CropSquare
                                style={{
                                    left: `${(cropArea.x / image.width) * 100}%`,
                                    top: `${(cropArea.y / image.height) * 100}%`,
                                    width: `${(cropArea.size / image.width) * 100}%`,
                                    height: `${(cropArea.size / image.height) * 100}%`
                                }}
                                onMouseDown={(e) => handleMouseDown(e, 'drag')}
                            >
                                <ResizeHandle 
                                    onMouseDown={(e) => {
                                        e.stopPropagation();
                                        handleMouseDown(e, 'resize');
                                    }} 
                                />
                            </CropSquare>
                        )}
                    </ImageWrapper>

                    {croppedPreview && (
                        <PreviewContainer>
                            <Caption>Preview (244x244):</Caption>
                            <PreviewImage src={croppedPreview} alt="Cropped preview" />
                        </PreviewContainer>
                    )}

                    <ButtonContainer>
                        {croppedPreview ? (
                            <>
                                <Button onClick={handleReCrop}>Re-crop</Button>
                                <Button onClick={handleTest}>Test Image</Button>
                            </>
                        ) : (
                            <Button onClick={handleCrop}>Crop Image</Button>
                        )}
                    </ButtonContainer>
                </ImageEditorContainer>
            )}
        </div>
    );
};

export default ImageUpload;