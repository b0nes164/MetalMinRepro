struct ScanParameters
{
    size: u32,
    unused_0: u32,
    unused_1: u32,
    unused_2: u32,
};

@group(0) @binding(0)
var<uniform> params : ScanParameters;

@group(0) @binding(1)
var<storage, read_write> scan_bump: u32;

@group(0) @binding(2)
var<storage, read_write> scan: array<u32>;

@group(0) @binding(3)
var<storage, read_write> errors: array<u32>;

const SPLIT_MEMBERS = 2u;
const BLOCK_DIM = 256u;

@compute @workgroup_size(BLOCK_DIM, 1, 1)
fn main(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(num_workgroups) griddim: vec3<u32>) {

    // Set scan states to NOT_READY
    for(var i = id.x; i < params.size * SPLIT_MEMBERS; i += griddim.x * BLOCK_DIM){
        scan[i] = 0u;
    }

    // Reset the atomic bump
    if(id.x == 0u){
        scan_bump = 0u;
    }

    // Clear the error counts
    for(var i = id.x; i < params.size * 4; i += griddim.x * BLOCK_DIM){
        errors[i] = 0u;
    }
}
