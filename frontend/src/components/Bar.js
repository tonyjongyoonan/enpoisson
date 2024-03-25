import React from 'react';

export default function Bar(props) {
    const {color, value, label} = props;

    const containerStyle = {
        width: 50,
        height: 572,
        backgroundColor: "white",
        border: "3px solid #353434",
        borderRadius: "4px",
    }

    const fillerStyle = {
        width: 44,
        height: `${value}%`,
        backgroundColor: color,
        textAlign: "center",
    }

    return(
        <div style={containerStyle}>
            <div style={fillerStyle}>
                <span>{label}</span>
            </div>
        </div>
    )
}