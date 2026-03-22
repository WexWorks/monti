#ifndef INTERIOR_LIST_GLSL
#define INTERIOR_LIST_GLSL

// ── Interior list for nested dielectric priority ─────────────────
// Sorted 2-slot interior list matching RTXPT's InteriorList model.
// Each slot packs: priority[31:28] | materialID[27:0].
// Priority in the high bits so integer sort gives highest-priority-first.
// Empty slots are 0. Priority 0 is remapped to kMaxSlotPriority (15)
// internally, so user priority 0 means "highest priority."

const uint kInteriorListSlots = 2u;
const uint kMaterialBits      = 28u;
const uint kPriorityBits      = 4u;
const uint kMaterialMask      = (1u << kMaterialBits) - 1u;
const uint kPriorityOffset    = kMaterialBits;
const uint kMaxSlotPriority   = (1u << kPriorityBits) - 1u;  // 15
const uint kNoMaterial        = 0xFFFFFFFFu;
const int  kMaxFalseIntersections = 4;

struct InteriorList {
    uint slots[kInteriorListSlots];
};

uint makeSlot(uint material_id, uint priority) {
    return (priority << kPriorityOffset) | (material_id & kMaterialMask);
}

bool isSlotActive(uint slot) {
    return slot != 0u;
}

uint getSlotPriority(uint slot) {
    return slot >> kPriorityOffset;
}

uint getSlotMaterialID(uint slot) {
    return slot & kMaterialMask;
}

void sortSlots(inout InteriorList list) {
    // 2-element sorting network: highest priority first
    if (list.slots[0] < list.slots[1]) {
        uint tmp = list.slots[0];
        list.slots[0] = list.slots[1];
        list.slots[1] = tmp;
    }
}

bool interiorListIsEmpty(InteriorList list) {
    return !isSlotActive(list.slots[0]);
}

uint getTopPriority(InteriorList list) {
    return getSlotPriority(list.slots[0]);
}

uint getTopMaterialID(InteriorList list) {
    return isSlotActive(list.slots[0]) ? getSlotMaterialID(list.slots[0]) : kNoMaterial;
}

uint getNextMaterialID(InteriorList list) {
    return isSlotActive(list.slots[1]) ? getSlotMaterialID(list.slots[1]) : kNoMaterial;
}

bool isTrueIntersection(InteriorList list, uint priority) {
    return priority == 0u || priority >= getTopPriority(list);
}

void handleIntersection(inout InteriorList list, uint material_id,
                        uint priority, bool entering) {
    // Remap priority 0 to highest (internally 0 is reserved for empty slots)
    if (priority == 0u) priority = kMaxSlotPriority;

    // Manually unrolled for 2 slots (matching RTXPT pattern)
    if (entering && list.slots[0] == 0u) {
        list.slots[0] = makeSlot(material_id, priority);
    } else if (!entering && isSlotActive(list.slots[0])
               && getSlotMaterialID(list.slots[0]) == material_id) {
        list.slots[0] = 0u;
    } else if (entering && list.slots[1] == 0u) {
        list.slots[1] = makeSlot(material_id, priority);
    } else if (!entering && isSlotActive(list.slots[1])
               && getSlotMaterialID(list.slots[1]) == material_id) {
        list.slots[1] = 0u;
    }

    sortSlots(list);
}

// Returns the materialID of the medium on the outside of the current interface.
// The caller must look up IOR from this materialID (or use 1.0 if kNoMaterial).
// material_id: the material being hit; entering: front-face hit.
uint getOutsideMaterialID(InteriorList list, uint material_id, bool entering) {
    if (entering) {
        // Entering: outside medium is the current top-of-stack (before insertion)
        return getTopMaterialID(list);
    } else {
        // Exiting: if this material is on top, outside is next; else outside is top
        uint top_id = getTopMaterialID(list);
        if (top_id == kNoMaterial) return kNoMaterial;
        if (getSlotMaterialID(list.slots[0]) == material_id) {
            return getNextMaterialID(list);
        }
        return top_id;
    }
}

#endif // INTERIOR_LIST_GLSL
