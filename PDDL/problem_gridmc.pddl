;; problem definition for Grid Minecraft

(define (problem ProblemGridMinecraft)

    (:domain GridMinecraft)
    
    (:objects
        key1 - key
        chest1 - chest
        sword1 - sword
        monster1 - monster
        gate1 - gate
    )
    
    (:init
        (keyExists key1)
        (chestExists chest1)
        (monsterExists monster1)
        (guardedGateExists monster1 gate1)
    )
    
    (:goal (and
        (actorReachedGate)
    ))
)
