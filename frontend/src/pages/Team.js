import React from 'react';
import './Team.css';

const Team = () => {
    const groupmates = [
        {
            name: 'Joseph Lee',
            image: 'joe.jpg',
            description: 'Research',
        },
        {
            name: 'Ishaan Lal',
            image: 'ishaan.jpg',
            description: 'Research',
        },
        {
            name: 'Andrew Jiang',
            image: 'andrew.jpg',
            description: 'Research',

        },
        {
            name: 'Nathaniel Lao',
            image: 'nate.jpg',
            description: 'Fullstack',

        },
        {
            name: 'Tony An',
            image: 'tony.jpg',
            description: 'Fullstack',

        }
    ];

    return (
        <div className="team-container">
            {groupmates.map((groupmate, index) => (
                <div key={index} className="groupmate-card">
                    <img src={groupmate.image} alt={groupmate.name} />
                    <h3>{groupmate.name}</h3>
                    <p>{groupmate.description}</p>
                </div>
            ))}
        </div>
    );
};

export default Team;
