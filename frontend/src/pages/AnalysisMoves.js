import React, { useState, useEffect, Fragment } from 'react';
import './AnalysisMoves.css';

export default function AnalysisMoves(props) {
    const {moves, current} = props;
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
        console.log(current)
    }, [current])


    return(
        <div className="movesContainer">
            {moveList.map(function(move) {
                return (
                    <>
                    <div className="move_no">
                        <p>{move["no"]}</p>
                    </div>
                    <div className="move_first">
                        <p> {move["first"]}</p>
                    </div>
                    {move["second"] ? 
                    <div className="move_second">
                        {move["second"]}
                    </div>
                    : <Fragment />}
                    </>
                )
                })}
        </div>
    )
}