<?php
/**
 * Plugin Name: art-rium — Yoast meta REST exposure
 * Description: Registers Yoast SEO post-meta keys (_yoast_wpseo_focuskw, _yoast_wpseo_metadesc, _yoast_wpseo_title) as REST-writable so the art-rium pipeline can set them via /wp/v2/posts. Yoast still owns indexable updates via its own update_post_meta hooks.
 * Version: 1.0.0
 * Author: art-rium
 *
 * Install: copy this file to wp-content/mu-plugins/art-rium-yoast-rest.php on the live site
 * (create the mu-plugins directory if it does not yet exist — must-use plugins auto-load,
 * no activation needed in wp-admin).
 */

if (!defined('ABSPATH')) {
    exit;
}

add_action('init', function () {
    $keys = [
        '_yoast_wpseo_focuskw'   => 'Yoast focus keyphrase',
        '_yoast_wpseo_metadesc'  => 'Yoast meta description',
        '_yoast_wpseo_title'     => 'Yoast SEO title override',
    ];

    foreach ($keys as $key => $description) {
        register_post_meta('post', $key, [
            'type'              => 'string',
            'description'       => $description,
            'single'            => true,
            'show_in_rest'      => true,
            'sanitize_callback' => 'sanitize_text_field',
            'auth_callback'     => function () {
                return current_user_can('edit_posts');
            },
        ]);
    }
});
