/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "windowAttentionPlugin/windowAttention.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/kernels/recentCache.h"

using namespace nvinfer1;
using namespace tensorrt_llm::kernels;
using namespace tensorrt_llm::common;
using tensorrt_llm::plugins::RecentCachePluginCreator;
using tensorrt_llm::plugins::RecentCachePlugin;

static const char* RECENTCACHE_PLUGIN_VERSION{"1"};
static const char* RECENTCACHE_PLUGIN_NAME{"RecentCache"};
PluginFieldCollection RecentCachePluginCreator::mFC{};
std::vector<nvinfer1::PluginField> RecentCachePluginCreator::mPluginAttributes;

RecentCachePlugin::RecentCachePlugin(int window_size, nvinfer1::DataType type)
    : window_size(window_size)
    , mType(type)
{
    TLLM_CHECK_WITH_INFO((getSMVersion() >= 80) || (mType != DataType::kBF16),
        "Unsupported data type, pre SM 80 GPUs do not support bfloat16");
}

// Parameterized constructor
RecentCachePlugin::RecentCachePlugin(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    read(d, window_size);
    read(d, mType);
    TLLM_CHECK(d == a + length);
    TLLM_CHECK_WITH_INFO((getSMVersion() >= 80) || (mType != DataType::kBF16), "Unsupported data type");
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* RecentCachePlugin::clone() const noexcept
{
    auto* plugin = new RecentCachePlugin(window_size, mType);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

nvinfer1::DimsExprs RecentCachePlugin::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    return inputs[outputIndex];
}

bool RecentCachePlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    TLLM_CHECK(0 <= pos && pos < 5);
    return (inOut[pos].type == mType) && (inOut[pos].format == TensorFormat::kLINEAR);
}

void RecentCachePlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
}

size_t RecentCachePlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    return 0;
}

int RecentCachePlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    // inputs
    //     past_key_value : [max_batch_size * max_beam_width, 2, num_heads, max_seqlen, hidden_dim_per_head]
    // outputs
    //     output [max_batch_size * max_beam_width, 2, num_heads, [:window_size], hidden_dim_per_head]

    // 获取tensor的维度信息
    int max_batch_size = inputDesc[0].dims.d[0];
    int max_beam_width = inputDesc[0].dims.d[1];
    int num_heads = inputDesc[0].dims.d[2];
    int max_seqlen = inputDesc[0].dims.d[3];
    int hidden_dim_per_head = inputDesc[0].dims.d[4];

    if (inputDesc[0].type == nvinfer1::DataType::kFLOAT)
    {
        const float* input = static_cast<const float*>(inputs[0]);
        float* output = static_cast<float*>(outputs[0]);
        invokesliceKernel<float>(input, output, max_batch_size, max_beam_width, num_heads, max_seqlen,
            hidden_dim_per_head, window_size, stream);
    }
    else if (inputDesc[0].type == nvinfer1::DataType::kHALF)
    {
        const half* input = static_cast<const half*>(inputs[0]);
        half* output = static_cast<half*>(outputs[0]);
        invokesliceKernel<half>(input, output, max_batch_size, max_beam_width, num_heads, max_seqlen,
            hidden_dim_per_head, window_size, stream);
    }
#ifdef ENABLE_BF16
    else if (inputDesc[0].type == nvinfer1::DataType::kBF16)
    {
        const __nv_bfloat16* input = static_cast<const __nv_bfloat16*>(inputs[0]);
        __nv_bfloat16* output = static_cast<__nv_bfloat16*>(outputs[0]);
        invokesliceKernel<__nv_bfloat16>(input, output, max_batch_size, max_beam_width, num_heads, max_seqlen,
            hidden_dim_per_head, window_size, stream);
    }
#endif
}

// IPluginV2Ext Methods
nvinfer1::DataType RecentCachePlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    assert(index == 0);
    return inputTypes[0];
}

// IPluginV2 Methods

const char* RecentCachePlugin::getPluginType() const noexcept
{
    return RECENTCACHE_PLUGIN_NAME;
}

const char* RecentCachePlugin::getPluginVersion() const noexcept
{
    return RECENTCACHE_PLUGIN_VERSION;
}

int RecentCachePlugin::getNbOutputs() const noexcept
{
    return 1;
}

int RecentCachePlugin::initialize() noexcept
{
    return 0;
}

void RecentCachePlugin::terminate() noexcept {}

size_t RecentCachePlugin::getSerializationSize() const noexcept
{
    return sizeof(window_size) + sizeof(mType);
}

void RecentCachePlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, window_size);
    write(d, mType);
    assert(d == a + getSerializationSize());
}

void RecentCachePlugin::destroy() noexcept
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

///////////////

RecentCachePluginCreator::RecentCachePluginCreator()
{
    // Fill PluginFieldCollection with PluginField arguments metadata
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("window_size", nullptr, PluginFieldType::kINT32, 2000));
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* RecentCachePluginCreator::getPluginName() const noexcept
{
    return RECENTCACHE_PLUGIN_NAME;
}

const char* RecentCachePluginCreator::getPluginVersion() const noexcept
{
    return RECENTCACHE_PLUGIN_VERSION;
}

const PluginFieldCollection* RecentCachePluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* RecentCachePluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    const PluginField* fields = fc->fields;
    int window_size;
    nvinfer1::DataType type;
    // Read configurations from each fields
    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "window_size"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            window_size = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "type_id"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            type = static_cast<nvinfer1::DataType>(*(static_cast<const nvinfer1::DataType*>(fields[i].data)));
        }
    }
    try
    {
        auto* obj = new RecentCachePlugin(window_size, type);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* RecentCachePluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call RecentCachePlugin::destroy()
    try
    {
        auto* obj = new RecentCachePlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}
