import React from 'react';

export default function Bar(props) {
    const {color, value, label, boardOrientation} = props;

    let containerStyle;
    let fillerStyle; 
    if (color !== "red") {
      if (boardOrientation) {
        containerStyle = {
          width: 50,
          height: 572,
          backgroundColor: "white",
          border: "3px solid #353434",
          borderRadius: "4px",
        }
        fillerStyle = {
          width: 44,
          height: `${value}%`,
          backgroundColor: color,
          textAlign: "center",
        }
      } else {
        containerStyle = {
          width: 50,
          height: 572,
          backgroundColor: "black",
          border: "3px solid #353434",
          borderRadius: "4px",
        }
        fillerStyle = {
          width: 44,
          height: `${value}%`,
          backgroundColor: "white",
          textAlign: "center",
        }
      }
      
    } else {
      containerStyle = {
        width: 50,
        height: 572,
        backgroundColor: "red",
        border: "3px solid #353434",
        borderRadius: "4px",
      }
      fillerStyle = {
        width: 44,
        height: `${value}%`,
        backgroundColor: "black",
        textAlign: "center",
      }
    }

    return(
        <div style={containerStyle}>
            <div style={fillerStyle}>
                <span style={{color: color === "red" ? "white" : boardOrientation ? (value < 3 ? "black" : "white") : (value < 3 ? "white" : "black")}}>{label}</span>
            </div>
        </div>
    )
}