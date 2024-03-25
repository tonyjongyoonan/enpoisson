import React, { useState, useEffect, useRef } from 'react';
import { useLocation } from 'react-router-dom';
import Chessboard from 'chessboardjsx';
import { Chess } from 'chess.js';
import AnalysisMoves from './AnalysisMoves';
import Bar from '../components/Bar';
import './Analyzed.css';

const Analyzed = () => {
  const [fen, setFen] = useState('start');
  const chess = useRef(new Chess());
  const moveIndex = useRef(0);
  const [moves, setMoves] = useState([]);
  const location = useLocation();
  const pgn = location.state.pgn;
  const moveIndexToFeedback = useRef({});
  const feedback = "";

  const getPgnMoves = (pgn) => {
    const newChess = new Chess();
    newChess.loadPgn(pgn);
    console.log(newChess.history());
    return newChess.history();
  };

  // TODO: tony can you create these three functions: 
  // nextMove(current) -- basically what you do with arrow right rn but takes in current and then updates chess board with newest move
  // updateBoard(move) -- takes in the index of a move and updates the board to that state in the game 
  // prevMove() -- basically what you do with arrow left rn and undoes a move

  useEffect(() => {
    setMoves(getPgnMoves(pgn)); // format: ['e4', 'e5', ..., 'Nf3', 'Nc6']
    const handleKeydown = (event) => {
      if (event.key === 'ArrowRight' && moveIndex.current < moves.length) {
        chess.current.move(moves[moveIndex.current]);
        setFen(chess.current.fen());
        moveIndex.current++;

        // if not cached, request for feedback
        if (!moveIndexToFeedback.current[moveIndex.current]) {
          const model_input = [];
          const history = chess.history({ verbose: true }).slice(0, moveIndex.current);
          const moves_made = moves.slice(0, moveIndex.current); // gets all moves so far
          for (let i = 0; i < moveIndex.length; i++) {
            model_input.append((history[i].from, moves_made.slice(Math.max(0, i - 16), i), history[i].color));
          }

          // get feedback from model
          // const feedback = model(model_input);
        } else {
          // display cached feedback
          feedback = moveIndexToFeedback.current[moveIndex.current];
        }
        feedback = "Feedback";

        // store feedback in hashmap
        moveIndexToFeedback.current[moveIndex.current] = feedback;

      } else if (event.key === 'ArrowLeft' && moveIndex.current > 0) {
        chess.current.undo();
        setFen(chess.current.fen());
        moveIndex.current--;

        // get cached feedback
        feedback = moveIndexToFeedback.current[moveIndex.current];
        
        // display feedback


      }
    };
    window.addEventListener('keydown', handleKeydown);
    return () => window.removeEventListener('keydown', handleKeydown);
  }, [pgn]);

  return (
    <div className="analysis-page-container">

      {/* <Progress.Line showInfo={false} strokeColor={"white"} vertical={true}/> */}
      <Bar color={"black"} value={50} label={"+0.0"}/>
      <Bar color={"blue"} value={80} label={"hard"}/>
      <div className="chessboard-container">
        <Chessboard position={fen} />
      </div>
      <div className="analysis-right-container">
        <div className="analysis-text-container">
          <h2>Analysis</h2>
          <p>TonySoTender vs. chmuina</p>
          <p>{"Here is the placeholder feedback--Andrew, when you have a chance to see this, can you please fix the formatting so that it's flexed properly for different sized feedback strings?"}</p>
        </div>
        <AnalysisMoves moves={moves} current={moveIndex.current}/>
      </div>
    </div>
  );
};  
export default Analyzed;