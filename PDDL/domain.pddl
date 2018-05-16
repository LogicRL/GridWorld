;; domain definition for Grid Minecraft environment

(define (domain GridMinecraft)

    (:requirements
    ;    :durative-actions
    ;    :equality
    ;    :negative-preconditions
    ;    :numeric-fluents
    ;    :object-fluents
        :typing
    )

    (:types key chest sword monster gate)

    ; (:constants
    ; 
    ; )

    (:predicates
        (keyExists ?key - key)
        (actorWithKey)
        (chestExists ?chest - chest)
        (actorWithSword)
        (monsterExists ?monster - monster)
        (freeGateExists ?gate - gate)
        (guardedGateExists ?monster - monster ?gate - gate)
        (actorReachedGate)
    )

    ; (:functions
    ; 
    ; )

    (:action grabKey
        ; grab a key
        :parameters (?key - key)
        :precondition (and
            (keyExists ?key)
            (not (actorWithKey))
        )
        :effect (and
            (actorWithKey)
            (not (keyExists ?key))
        )
    )

    (:action openChest
        ; open a chest to obtain a magic sword
        :parameters (?key - key ?chest - chest)
        :precondition (and
            (chestExists ?chest)
            (actorWithKey)
            (not (actorWithSword))
        )
        :effect (and
            (not (actorWithKey))
            (not (chestExists ?chest))
            (actorWithSword)
        )
    )

    (:action killMonster
        ; kill a monster with a magic sword
        :parameters (?monster - monster)
        :precondition (and
            (monsterExists ?monster)
            (actorWithSword)
        )
        :effect (and
            (not (monsterExists ?monster))
        )
    )

    (:action goToFreeGate
        ; go the a free gate without being guarded by a monster
        :parameters (?gate - gate)
        :precondition (and
            (freeGateExists ?gate)
            (not (actorReachedGate))
        )
        :effect (and
            (actorReachedGate)
        )
    )

    (:action goToGuardedGate
        ; go the a gate guarded by a monster
        :parameters (?monster - monster ?gate - gate)
        :precondition (and
            (guardedGateExists ?monster ?gate)
            (not (monsterExists ?monster))
            (not (actorReachedGate))
        )
        :effect (and
            (actorReachedGate)
        )
    )
)
