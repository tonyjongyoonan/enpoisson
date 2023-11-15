import React from 'react'
import '../styles/Board.css'

function Board() {

    const getTileColor = (i,j) => {
        let c = 'tile'
        c += (i + j) % 2 === 0 ? '-light' : '-dark'
        return c
    }



    const ranks = Array(8).fill().map((_, i) => 8 - i)
    const files = Array(8).fill().map((_, i) => String.fromCharCode(97 + i))
    
    return <div className="board">

        <div className="tiles">
            {ranks.map((rank, i) => 
                files.map((file, j) => 
                    <div key={file + '-' + rank} className={getTileColor(i, j)}>{file}{rank}</div>
                )
            )}
        </div>
    </div>
}

export default Board
