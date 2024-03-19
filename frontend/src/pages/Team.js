import React from 'react';
import './Team.css';
import joe from '../assets/joe.png';
import ishaan from '../assets/ishaan.png';
import andrew from '../assets/andrew.png';
import nate from '../assets/nate.png';
import tony from '../assets/tony.png';

const Team = () => {
    const groupmates = [
        {
            name: 'Joseph Lee',
            image: joe,
            description: 'Research',
        },
        {
            name: 'Ishaan Lal',
            image: ishaan,
            description: 'Research',
        },
        {
            name: 'Andrew Jiang',
            image: andrew,
            description: 'Research',

        },
        {
            name: 'Nathaniel Lao',
            image: nate,
            description: 'Fullstack',

        },
        {
            name: 'Tony An',
            image: tony,
            description: 'Fullstack',

        }
    ];

    return (
        <div className="team-container">
            {groupmates.map((groupmate, index) => (
                <div key={index} className="groupmate-card">
                    <img src={groupmate.image} alt={groupmate.name} style={{ width: '100px', height: '100px' }} />
                    <h3>{groupmate.name}</h3>
                    <p>{groupmate.description}</p>
                </div>
            ))}
        </div>
    );
};

export default Team;
