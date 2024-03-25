import React, { useState, useEffect, Fragment } from 'react';
import './AnalysisMoves.css';

export default function AnalysisMoves(props) {
    const {moves, index, updateMove} = props;
    const [moveList, setMoveList] = useState([]);
    useEffect(() => {
        let num_rows = Math.floor((moves.length + 1)/ 2);
        let all_moves = [];
        for (let i = 1; i < num_rows + 1; i++) {
            let current_turn = {};
            current_turn["no"] = i;
            current_turn["first"] = moves[i * 2 - 2]; 
            if (moves.length > i * 2 - 1) {
                current_turn["second"] = moves[i * 2 - 1];
            }
            all_moves.push(current_turn);
        }
        setMoveList(all_moves);
    }, [moves])

    useEffect(() => {
        console.log(index)
    }, [index])


    return(
        <div className="movesContainer">
            {moveList.map(function(move) {
                return (
                    <>
                    <div className="move_no">
                        <p>{move["no"]}</p>
                    </div>
                    <div className="moveFirst" onClick={(e) => updateMove(move["no"] * 2 - 1)}>
                        <p style={{ color : move["no"] * 2 - 1 === index ? "yellow" : "white"}}>{move["first"]}</p>
                    </div>
                    {move["second"] ? 
                    <div className="moveSecond" onClick={(e) => updateMove(move["no"] * 2)}>
                        <p style={{ color : move["no"] * 2 === index ? "yellow" : "white"}}>{move["second"]}</p>
                    </div>
                    : <Fragment />}
                    </>
                )
                })}
        </div>
    )
}