// const { OnnxModel } = require("./build/Release/onnx_node_gpu.node");
import { Tokenizer } from "./src/lib/tokenizer.js";

import { createRequire } from 'module';
const require = createRequire(import.meta.url);

const { OnnxModel } = require('./build/Release/onnx_node_gpu.node');

const textEncoder = new OnnxModel("text_encoder/model.onnx")
function extendArray(arr, length) {
  return arr.concat(Array(length - arr.length).fill(0));
}
console.log('textEncoder', textEncoder)
function encodePrompt (prompt, tokenizer, textEncoder) {
  const tokens = tokenizer.encode(prompt)
  const paddedTokens = Int32Array.from(extendArray([49406, ...tokens, 49407], 77))

  return textEncoder.runInference({
    inputs: {
      input_ids: { data: paddedTokens, shape: [1, 77] },
    },
    outputs: ['last_hidden_state'],
  })
}

async function getPromptEmbeds (prompt: string, negativePrompt: string|undefined) {
  const tokenizer = new Tokenizer()
  const promptEmbeds = encodePrompt(prompt, tokenizer, textEncoder)
  const negativePromptEmbeds = encodePrompt(negativePrompt || '', tokenizer, textEncoder)

  return [...negativePromptEmbeds, promptEmbeds]
}


// use an async context to call onnxruntime functions.
async function inference () {
  const guidanceScale = 9
  const width = 512
  const height = 512

  const promptEmbeds = await getPromptEmbeds('a dog on a lawn with the eifel tower in the background', '')
  const latentShape = [1, 4, width / 8, height / 8]

  const config = {
    "beta_end": 0.012,
    "beta_schedule": "scaled_linear",
    "beta_start": 0.00085,
    "clip_sample": false,
    "num_train_timesteps": 1000,
    "prediction_type": "epsilon",
    "set_alpha_to_one": false,
    "skip_prk_steps": true,
    "steps_offset": 1,
    "trained_betas": null
  }
  const scheduler = new PNDMScheduler(
      config,
      config.num_train_timesteps,
      config.beta_start,
      config.beta_end,
      config.beta_schedule,
  )
  await scheduler.setAlphasCumprod()
  scheduler.setTimesteps(29)

  // Generate an array of random values
  let latents = tf.randomNormal(latentShape, undefined, undefined, 'float32')
  const unet = await InferenceSession.create('./public/sd2_1base/unet/model.onnx', sessionOption);

  for (const step of await scheduler.timesteps.data()) {
    console.log('step', step, 'of')
    const timestep = new Tensor('float32', [step])

    const latentInput = tf.concat([latents, latents])
    let noise = await unet.run({ sample: latentInput, timestep, encoder_hidden_states: promptEmbeds })

    let noisePred = Object.values(noise)[0].data
    // let noisePredDims = Object.values(noise)[0].dims as number[]
    // const sample = tf.tensor(noisePred, Object.values(noise)[0].dims as number[])

    const len = Object.values(noise)[0].data.length / 2
    const [noisePredUncond, noisePredText] = [
      tf.tensor(noisePred.slice(0, len), latentShape, 'float32'),
      tf.tensor(noisePred.slice(len, len * 2), latentShape, 'float32'),
    ]
    noisePred = noisePredUncond.add(noisePredText.sub(noisePredUncond).mul(guidanceScale))

    const schedulerOutput = scheduler.step(
        noisePred,
        step,
        latents,
    )
    latents = schedulerOutput
    // console.log('latents', latents.shape, await latents.data())
  }

  latents = latents.mul(tf.tensor(1).div(0.18215))
  // console.log('latents', latents.shape, await latents.data())
  const vae = await InferenceSession.create('./public/sd2_1base/vae_decoder/model.onnx', sessionOption)
  const image = await vae.run({ latent_sample: new Tensor('float32', await latents.data(), [1, 4, width / 8, height / 8]) })

  let imageData = tf.tensor(Object.values(image)[0].data, [3, width, height], 'float32')
  imageData = imageData.div(2).add(0.5).transpose([1, 2, 0])
  imageData = imageData.mul(255).round().clipByValue(0, 255).cast('int32')
  await saveRGBFloat32ArrayToJPG(await imageData.data(), width, height, './output.jpg')

  // Save Buffer as JPG file
  // return sharp(buffer, { raw: { width, height, channels: 3 } })
  //   .jpeg()
  //   .toFile('output.jpg')
  // tf.node.encodeJpeg(imageData.transpose([1, 2, 0])).then((f) => {
  //   fs.writeFileSync("simple.jpg", f);
  //   console.log("Basic JPG 'simple.jpg' written");
  // })

  // await saveRGBFloat32ArrayToJPG(await imageData.data(), width, height, './output.jpg')
}

console.log('result', inference())
