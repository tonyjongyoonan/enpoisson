import React from 'react';
import poster from '../assets/poster.png';
import './Tech.css';

const Tech = () => {
    const info = [
        {
            image: poster
        },
    ];

    return (
        <div>
            <div className="poster-container">
                {info.map((poster, index) => (
                    <div key={index} className="poster">
                        <img src={poster.image} alt={poster.name} className="poster-img" />
                        <h3>{poster.name}</h3>
                    </div>
                ))}
            </div>
            <div className="tech-description">
                <p>In our model, we measure “move-matching accuracy”, how often En Poisson's predicted move</p>
                <p>is the same as the human move played in real online games.</p>
                <p style={{ marginBottom: '1.5rem' }}></p>
                <p>As a comparison, we looked at how depth-limited Stockfish and Maia do on the same prediction task.</p>
                <p style={{ marginBottom: '1.5rem' }}></p>
                <p> The median accuracy for Stockfish is at 0.35, Maia at 0.475, and for En Poisson, at 0.60.</p>
                <p style={{ marginBottom: '1.5rem' }}></p>
                <p>This makes En Poisson's Engine the State-of-the-Art (SOTA) model.</p>
                </div>
            </div>
    );
};
export default Tech;