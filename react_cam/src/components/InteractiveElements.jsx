import React, { useEffect, useState } from 'react';
import './InteractiveElements.css';

const InteractiveElements = () => {
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  const [trails, setTrails] = useState([]);
  const [stars, setStars] = useState([]);

  useEffect(() => {
    // Create initial stars
    const initialStars = Array.from({ length: 15 }, (_, i) => ({
      id: i,
      x: Math.random() * window.innerWidth,
      y: Math.random() * window.innerHeight,
      size: Math.random() * 15 + 10,
    }));
    setStars(initialStars);

    // Mouse move handler
    const handleMouseMove = (e) => {
      const x = (e.clientX / window.innerWidth) * 100;
      const y = (e.clientY / window.innerHeight) * 100;
      setMousePosition({ x, y });
      
      // Add new trail particle
      setTrails(prevTrails => [...prevTrails, {
        id: Date.now(),
        x: e.clientX,
        y: e.clientY
      }].slice(-20)); // Keep only last 20 particles
    };

    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

  const getParallaxStyle = (strength) => ({
    transform: `translate(
      ${(mousePosition.x - 50) * strength}px,
      ${(mousePosition.y - 50) * strength}px
    )`
  });

  return (
    <div className="interactive-container">
      {/* Floating Characters */}
      <div className="floating-character character1" style={getParallaxStyle(-0.1)}>🦒</div>
      <div className="floating-character character2" style={getParallaxStyle(0.15)}>🎈</div>
      <div className="floating-character character3" style={getParallaxStyle(-0.2)}>🦋</div>
      <div className="floating-character character4" style={getParallaxStyle(0.1)}>✨</div>

      {/* Interactive Stars */}
      {stars.map(star => (
        <div
          key={star.id}
          className="interactive-star"
          style={{
            left: `${star.x}px`,
            top: `${star.y}px`,
            width: `${star.size}px`,
            height: `${star.size}px`,
            ...getParallaxStyle(0.05)
          }}
        />
      ))}

      {/* Mouse Trail */}
      {trails.map(trail => (
        <div
          key={trail.id}
          className="trail-particle"
          style={{
            left: `${trail.x}px`,
            top: `${trail.y}px`
          }}
        />
      ))}

      {/* Parallax Background Elements */}
      <div className="parallax-layer layer1" style={getParallaxStyle(-0.05)}>🌸</div>
      <div className="parallax-layer layer2" style={getParallaxStyle(0.08)}>🌺</div>
      <div className="parallax-layer layer3" style={getParallaxStyle(-0.12)}>🍃</div>
    </div>
  );
};

export default InteractiveElements; 