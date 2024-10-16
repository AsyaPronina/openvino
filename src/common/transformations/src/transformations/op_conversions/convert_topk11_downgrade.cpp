// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_topk11_downgrade.hpp"

#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <openvino/opsets/opset11.hpp>
#include <openvino/opsets/opset3.hpp>

#include "itt.hpp"

ov::pass::ConvertTopK11ToTopK3::ConvertTopK11ToTopK3() {
    MATCHER_SCOPE(ConvertTopK11ToTopK3);

    const auto topk_v11_pattern = pattern::wrap_type<opset11::TopK>();

    const matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto topk_v11 = std::dynamic_pointer_cast<opset11::TopK>(m.get_match_root());
        if (!topk_v11 || topk_v11->get_stable() || transformation_callback(topk_v11)) {
            return false;
        }

        // downgrade only if the stable sort is NOT required

        const auto topk_v3 = std::make_shared<opset3::TopK>(topk_v11->input_value(0),
                                                            topk_v11->input_value(1),
                                                            topk_v11->get_axis(),
                                                            topk_v11->get_mode(),
                                                            topk_v11->get_sort_type(),
                                                            topk_v11->get_index_element_type());

        topk_v3->set_friendly_name(topk_v11->get_friendly_name());
        copy_runtime_info(topk_v11, topk_v3);
        replace_node(topk_v11, topk_v3);

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(topk_v11_pattern, matcher_name);
    register_matcher(m, callback);
}
