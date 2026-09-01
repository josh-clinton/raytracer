// bvh.h - Bounding Volume Hierarchy acceleration structure.
//
// Without this, tracing a ray against N objects is O(N) per ray. A BVH
// recursively splits the scene into a binary tree of bounding boxes so a
// ray that misses a subtree's box skips every object inside it, turning
// intersection into roughly O(log N). This is the single most important
// "performance optimization" concept in the whole project, and it's the
// same core idea (spatial acceleration structures) behind the BVHs that
// real-time hardware ray tracing (DXR/RT cores) traverses on GPU.
#pragma once

#include <algorithm>
#include <memory>
#include <vector>

#include "hittable.h"
#include "hittable_list.h"

class bvh_node : public hittable {
public:
    explicit bvh_node(hittable_list list) : bvh_node(list.objects, 0, list.objects.size()) {}

    bvh_node(std::vector<std::shared_ptr<hittable>>& objects, size_t start, size_t end) {
        bbox = aabb::empty_box();
        for (size_t i = start; i < end; i++) bbox = aabb(bbox, objects[i]->bounding_box());

        int axis = bbox.longest_axis();
        auto comparator = [axis](const std::shared_ptr<hittable>& a,
                                  const std::shared_ptr<hittable>& b) {
            return a->bounding_box().axis_interval(axis).min <
                   b->bounding_box().axis_interval(axis).min;
        };

        size_t object_span = end - start;

        if (object_span == 1) {
            left = right = objects[start];
        } else if (object_span == 2) {
            left = objects[start];
            right = objects[start + 1];
        } else {
            std::sort(objects.begin() + start, objects.begin() + end, comparator);
            size_t mid = start + object_span / 2;
            left = std::make_shared<bvh_node>(objects, start, mid);
            right = std::make_shared<bvh_node>(objects, mid, end);
        }
    }

    bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
        if (!bbox.hit(r, ray_t)) return false;

        bool hit_left = left->hit(r, ray_t, rec);
        bool hit_right = right->hit(r, interval(ray_t.min, hit_left ? rec.t : ray_t.max), rec);

        return hit_left || hit_right;
    }

    aabb bounding_box() const override { return bbox; }

private:
    std::shared_ptr<hittable> left;
    std::shared_ptr<hittable> right;
    aabb bbox;
};
