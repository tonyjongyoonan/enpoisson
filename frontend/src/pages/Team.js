import React from 'react';
import joe from '../assets/joe.png';
import ishaan from '../assets/ishaan.png';
import andrew from '../assets/andrew.png';
import nate from '../assets/nate.png';
import tony from '../assets/tony.png';
import GitHubIcon from '@mui/icons-material/GitHub';
import './Team.css';

const Team = () => {
    const groupmates = [
        {
            name: 'Ishaan Lal',
            image: ishaan,
            description: 'LLM',
            github: 'https://github.com/ishlal'
        },
        {
            name: 'Andrew Jiang',
            image: andrew,
            description: 'LLM',
            github: 'https://github.com/aJayz54'
        },
        {
            name: 'Joseph Lee',
            image: joe,
            description: 'Engine',
            github: 'https://github.com/jiosephlee'
        },
        {
            name: 'Nathaniel Lao',
            image: nate,
            description: 'Fullstack',
            github: 'https://github.com/nlao1'
        },
        {
            name: 'Tony An',
            image: tony,
            description: 'Fullstack',
            github: 'https://github.com/tonyjongyoonan'
        }
    ];

    return (
        <div>
            <div className="team-container">
                {groupmates.map((groupmate, index) => (
                    <div key={index} className="groupmate-card">
                        <img src={groupmate.image} alt={groupmate.name} style={{ width: '100px', height: '100px' }} />
                        <h3>{groupmate.name}</h3>
                        <p>{groupmate.description}</p>
                        <a href={groupmate.github} target="_blank" rel="noopener noreferrer">
                            <GitHubIcon className="github" sx={{ color: "white" }} />
                        </a>
                    </div>
                ))}
            </div>
            <div className="team-description">
                <p>We are a group of students at the University of Pennsylvania</p>
                <p>Interested in the intersection of Chess education and AI/ML.</p>
                <p style={{ marginBottom: '1.5rem' }}></p>
                <a href="tech" className="large-link">En Poisson</a> is a model that recommends powerful,
                <p>human-like moves, avoiding the chaos and confusion</p>
                <p>typically associated with Stockfish and other engines.</p>
                </div>
            </div>
    );
};
export default Team;